from datasets import load_dataset, DatasetDict
from collections import Counter
from tqdm import tqdm
import spacy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from functools import partial  # <-- 引入 partial
import math

# 定义特殊标记及其索引
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# 确保特殊标记的顺序与索引一致
special_symbols = ['<unk>', '<pad>', '<s>', '</s>']


def get_tokenizers():
    """直接加载 Spacy 分词器模型用于英语和德语。"""
    print("正在加载 Spacy 分词器模型...")
    try:
        en_tokenizer = spacy.load('en_core_web_sm')
        de_tokenizer = spacy.load('de_core_news_sm')
        print("Spacy 模型加载成功！")
        return en_tokenizer, de_tokenizer
    except OSError:
        print("未找到 Spacy 模型。请运行以下命令下载:")
        print("python -m spacy download en_core_web_sm")
        print("python -m spacy download de_core_news_sm")
        raise


def load_raw_data(dataset_name: str, language_pair: str) -> DatasetDict:
    """从 Hugging Face Hub 加载指定的原始数据集和语言对。"""
    print(f"开始加载 {dataset_name} 数据集, 语言对: {language_pair}...")

    raw_datasets = load_dataset(
        dataset_name,
        language_pair,
        trust_remote_code=True)

    print("数据集加载成功！")
    return raw_datasets


class Vocab:
    """一个简单的词典类，用于管理 token 到 index 的映射。"""

    def __init__(self, token_counter, min_freq=1):
        self.stoi = {}
        self.itos = {}
        for i, symbol in enumerate(special_symbols):
            self.stoi[symbol] = i
            self.itos[i] = symbol
        for token, freq in token_counter.items():
            if freq >= min_freq:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        """让 Vocab 可以像字典一样使用 vocab['<pad>']"""
        return self.stoi[token]

    def text_to_ids(self, text, tokenizer):
        doc = tokenizer(text)
        return [self.stoi.get(str(token), UNK_IDX) for token in doc]

    def ids_to_text(self, ids):
        return " ".join([self.itos.get(idx, '<unk>') for idx in ids])


def build_vocabs_and_get_stats(raw_datasets, en_tokenizer, de_tokenizer):
    train_data = raw_datasets['train']

    def tokenize_batch(batch):
        source_texts = [item['en'] for item in batch['translation']]
        target_texts = [item['de'] for item in batch['translation']]

        en_docs = en_tokenizer.pipe(source_texts)
        de_docs = de_tokenizer.pipe(target_texts)

        # 确保 en_docs 对应 source_tokens (英语), de_docs 对应 target_tokens (德语)
        batch['source_tokens'] = [[str(token) for token in doc] for doc in en_docs]
        batch['target_tokens'] = [[str(token) for token in doc] for doc in de_docs]
        return batch

    print("开始使用 .map() 高效分词 (这步只在第一次运行时较慢)...")
    tokenized_train_data = train_data.map(
        tokenize_batch, batched=True, num_proc=4, batch_size=1000
    )
    print("分词完成！结果已被缓存。")

    de_token_counter = Counter()
    en_token_counter = Counter()
    max_len_source = 0
    max_len_target = 0

    print("开始从已分词数据中构建词典...")
    for item in tqdm(tokenized_train_data):
        # 这里的 source/target 对应 tokenize_batch 中的 source/target
        source_tokens = item['source_tokens']
        target_tokens = item['target_tokens']

        en_token_counter.update(source_tokens)
        de_token_counter.update(target_tokens)

        max_len_source = max(max_len_source, len(source_tokens) + 1)  # 只加 EOS
        max_len_target = max(max_len_target, len(target_tokens) + 1)  # 只加 EOS

    en_vocab = Vocab(en_token_counter, min_freq=2)
    de_vocab = Vocab(de_token_counter, min_freq=2)

    print("\n词典构建与统计完成！")
    print(f"源语言 (en) 词典大小: {len(en_vocab)}")
    print(f"目标语言 (de) 词典大小: {len(de_vocab)}")
    print(f"源语言最长句子长度 (含EOS): {max_len_source}")
    print(f"目标语言最长句子长度 (含EOS): {max_len_target}")

    return en_vocab, de_vocab, max_len_source, max_len_target


class TranslationDataset(Dataset):
    """一个自定义的 PyTorch Dataset，用于翻译任务。"""

    def __init__(self, data_split, en_vocab, de_vocab, en_tokenizer, de_tokenizer):
        super().__init__()
        self.data = data_split
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab
        self.en_tokenizer = en_tokenizer
        self.de_tokenizer = de_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]['translation']
        source_text = item['en']
        target_text = item['de']

        source_ids = self.en_vocab.text_to_ids(source_text, self.en_tokenizer)
        target_ids = self.de_vocab.text_to_ids(target_text, self.de_tokenizer)

        # Dataset 只负责加上结束符，返回未填充的张量
        source_ids_with_eos = source_ids + [EOS_IDX]
        target_ids_with_eos = target_ids + [EOS_IDX]

        return torch.tensor(source_ids_with_eos), torch.tensor(target_ids_with_eos)


# --- 关键修正：重写整个 collate_fn ---
def collate_fn(batch, pad_idx, sos_idx):
    """
    自定义的 collate_fn，遵循"先处理、后填充"的原则。
    它接收一个由 Dataset 的 __getitem__ 返回的元组组成的列表。
    """
    source_batch, target_batch = [], []
    for (src_item, tgt_item) in batch:
        source_batch.append(src_item)
        target_batch.append(tgt_item)  # tgt_item 现在是原始的 "label"

    # 1. 填充源序列 (Encoder Input)
    encoder_input = pad_sequence(source_batch, batch_first=True, padding_value=pad_idx)

    # 2. 填充目标序列 (Decoder Labels)
    decoder_label = pad_sequence(target_batch, batch_first=True, padding_value=pad_idx)

    # 3. 创建解码器输入 (Decoder Input)
    decoder_inputs = []
    for item in target_batch:  # 遍历未填充的 target 列表
        # item 的形式是 [token_ids, EOS_IDX]
        # 我们需要创建 [SOS_IDX, token_ids]
        dec_input = torch.cat([torch.tensor([sos_idx]), item[:-1]], dim=0)
        decoder_inputs.append(dec_input)

    decoder_input = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_idx)

    return encoder_input, decoder_input, decoder_label


# def get_dataloaders(config):
#     """完整的数据加载和预处理流程。"""
#     raw_datasets = load_raw_data(config['dataset_name'], config['language_pair'])
#     en_tokenizer, de_tokenizer = get_tokenizers()
#     en_vocab, de_vocab, max_src, max_tgt = build_vocabs_and_get_stats(
#         raw_datasets, en_tokenizer, de_tokenizer
#     )
#
#     train_dataset = TranslationDataset(
#         raw_datasets['train'], en_vocab, de_vocab, en_tokenizer, de_tokenizer
#     )
#     val_dataset = TranslationDataset(
#         raw_datasets['validation'], en_vocab, de_vocab, en_tokenizer, de_tokenizer
#     )
#
#     # --- 关键修正 ---
#     # 使用 functools.partial 将固定的 pad_idx 和 sos_idx 参数传入 collate_fn
#     collate_with_indices = partial(collate_fn, pad_idx=PAD_IDX, sos_idx=SOS_IDX)
#
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         collate_fn=collate_with_indices,
#         num_workers=4,
#         pin_memory=True
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         collate_fn=collate_with_indices,
#         num_workers=4,
#         pin_memory=True
#     )
#
#     return train_dataloader, val_dataloader, en_vocab, de_vocab, max_src, max_tgt

def get_dataloaders(config):
    """
    完整的数据加载和预处理流程。
    现在支持使用训练集的一个子集进行快速调试。
    """
    # 1. 加载原始数据
    raw_datasets = load_raw_data(config['dataset_name'], config['language_pair'])

    # --- 新增逻辑：获取训练集的一个子集 ---
    train_split = raw_datasets['train']
    train_subset_ratio = config.get('train_subset_ratio', 1.0)  # 默认为1.0 (使用全部数据)

    if train_subset_ratio < 1.0:
        print(f"\n[INFO] 使用训练集 {train_subset_ratio * 100:.0f}% 的数据进行训练...")
        num_samples = math.ceil(len(train_split) * train_subset_ratio)

        # .select() 方法可以高效地选择一个索引范围
        # 为了确保每次选择的子集都一样，我们不进行随机抽样，而是取前 N 个样本
        train_split = train_split.select(range(num_samples))
        print(f"[INFO] 子集大小: {len(train_split)} 条样本。")

    # 2. 获取分词器
    en_tokenizer, de_tokenizer = get_tokenizers()

    # 3. 构建词典并获取统计信息
    # 注意：词典仍然在【完整】的训练集上构建，这是最佳实践！
    # 这样可以确保即使在小数据集上训练，模型也认识所有可能的词。
    print("\n[INFO] 在【完整】训练集上构建词典...")
    en_vocab, de_vocab, max_src, max_tgt = build_vocabs_and_get_stats(
        raw_datasets, en_tokenizer, de_tokenizer
    )

    # 4. 为训练集和验证集创建 Dataset 实例
    # --- 修改点：使用我们切片后的 train_split ---
    train_dataset = TranslationDataset(
        train_split,  # <-- 使用子集
        en_vocab,
        de_vocab,
        en_tokenizer,
        de_tokenizer
    )
    val_dataset = TranslationDataset(
        raw_datasets['validation'],
        en_vocab,
        de_vocab,
        en_tokenizer,
        de_tokenizer
    )

    # 5. 创建 DataLoader 实例
    collate_with_indices = partial(collate_fn, pad_idx=PAD_IDX, sos_idx=SOS_IDX)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,  # 在子集上打乱是必要的
        collate_fn=collate_with_indices,
        num_workers=config.get('num_workers', 0),  # 从 config 获取，默认为0 (安全)
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_with_indices,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )

    return train_dataloader, val_dataloader, en_vocab, de_vocab, max_src, max_tgt


# ==============================================================================
#                              测试用例
# ==============================================================================
if __name__ == '__main__':
    # 1. 定义一个简单的配置
    config = {
        'dataset_name': 'iwslt2017',
        'language_pair': 'iwslt2017-en-de',
        'batch_size': 3,  # 使用小 batch size 方便观察
    }

    print("--- 开始测试数据加载流程 ---")

    # 2. 获取 DataLoader 和词典
    # 我们只测试训练集，所以只取第一个返回的 loader
    # 为了测试，我们将 shuffle 设置为 False，确保每次运行结果一致
    try:
        train_loader, _, en_vocab, de_vocab, _, _ = get_dataloaders(config)

        # 为了可复现的测试，我们重新创建一个不打乱的 loader
        test_dataset = train_loader.dataset
        collate_with_indices = partial(collate_fn, pad_idx=PAD_IDX, sos_idx=SOS_IDX)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_with_indices
        )

        # 3. 从 loader 中取出一个批次的数据
        print("\n--- 从 DataLoader 中取出一个批次 ---")
        encoder_input, decoder_input, decoder_labels = next(iter(test_loader))

        # 4. 打印批次中张量的形状
        print(f"\nEncoder Input Shape: {encoder_input.shape}")
        print(f"Decoder Input Shape: {decoder_input.shape}")
        print(f"Decoder Labels Shape: {decoder_labels.shape}")

        # 5. 将 ID 转换回文本并打印，以验证逻辑
        print("\n--- 逐条解码批次中的数据 ---")
        for i in range(config['batch_size']):
            print(f"\n--- Sample {i + 1} ---")

            # 从批次中取出第 i 条数据，并转换为 Python 列表
            enc_ids = encoder_input[i].tolist()
            dec_in_ids = decoder_input[i].tolist()
            dec_lab_ids = decoder_labels[i].tolist()

            # 使用词典进行解码
            enc_text = en_vocab.ids_to_text(enc_ids)
            dec_in_text = de_vocab.ids_to_text(dec_in_ids)
            dec_lab_text = de_vocab.ids_to_text(dec_lab_ids)

            print(f"  Encoder Input : {enc_text}")
            print(f"  Decoder Input : {dec_in_text}")
            print(f"  Decoder Labels: {dec_lab_text}")

        print("\n--- 预期结果分析 ---")
        print("1. Encoder Input  应为【英语】句子，以 `</s>` 结尾，不足处用 `<pad>` 填充。")
        print("2. Decoder Input  应为【德语】句子，以 `<s>` 开头，不含结尾的 `</s>`，不足处用 `<pad>` 填充。")
        print("3. Decoder Labels 应为【德语】句子，不含开头的 `<s>`，以 `</s>` 结尾，不足处用 `<pad>` 填充。")
        print("4. Decoder Input 和 Decoder Labels 的内容应该是错开一位的，且长度一致。")
        print("\n--- 测试完成 ---")

    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()