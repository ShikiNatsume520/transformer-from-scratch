from datasets import load_dataset, DatasetDict
from collections import Counter
from tqdm import tqdm
import spacy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# 定义特殊标记及其索引
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# 确保特殊标记的顺序与索引一致
special_symbols = ['<unk>', '<pad>', '<s>', '</s>']


def get_tokenizers():
    """
    直接加载 Spacy 分词器模型用于英语和德语。
    """
    print("正在加载 Spacy 分词器模型...")
    try:
        # spacy.load() 会加载完整的模型并返回一个可调用的对象
        en_tokenizer = spacy.load('en_core_web_sm')
        de_tokenizer = spacy.load('de_core_news_sm')
        print("Spacy 模型加载成功！")
        return en_tokenizer, de_tokenizer
    except OSError:
        print("未找到 Spacy 模型。请运行以下命令下载:")
        print("python -m spacy download en_core_web_sm")
        print("python -m spacy download de_core_news_sm")
        # 抛出异常，让程序停止，而不是返回 None
        raise

def load_raw_data(dataset_name: str, language_pair: str) -> DatasetDict:
    """
    从 Hugging Face Hub 加载指定的原始数据集和语言对。

    Args:
        dataset_name (str): 数据集的名称 (例如, "iwslt2017")。
        language_pair (str): 数据集中特定的子集或语言对
                             (例如, "iwslt2017-en-de")。

    Returns:
        DatasetDict: 一个包含 'train', 'validation', 'test' 三个分割的数据集字典。
    """

    print(f"开始加载 {dataset_name} 数据集, 语言对: {language_pair}...")

    # 使用传入的参数加载数据集
    # trust_remote_code=True 仍然是必需的
    raw_datasets = load_dataset(
        dataset_name,
        language_pair,
        trust_remote_code=True
    )

    print("数据集加载成功！")

    return raw_datasets


class Vocab:
    """
    一个简单的词典类，用于管理 token 到 index 的映射。
    """

    def __init__(self, token_counter, min_freq=1):
        self.stoi = {}  # string-to-index
        self.itos = {}  # index-to-string

        # 首先添加特殊标记
        for i, symbol in enumerate(special_symbols):
            self.stoi[symbol] = i
            self.itos[i] = symbol

        # 从计数器中添加满足最小频率的单词
        for token, freq in token_counter.items():
            if freq >= min_freq:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self):
        return len(self.itos)

    def text_to_ids(self, text, tokenizer):
        """将文本字符串转换为 token ID 列表"""
        # tokenizer(text) 返回的是一个 Doc 对象，里面是 Token 对象
        doc = tokenizer(text)

        # 关键修正：在查找字典前，将每个 token 对象转换为它的文本字符串
        return [self.stoi.get(str(token), UNK_IDX) for token in doc]

    def ids_to_text(self, ids):
        """将 token ID 列表转换为文本字符串"""
        # 这个方法应该没问题，因为 self.itos 的值就是字符串
        return " ".join([self.itos.get(idx, '<unk>') for idx in ids])


def build_vocabs_and_get_stats(raw_datasets, en_tokenizer, de_tokenizer):
    train_data = raw_datasets['train']

    # --- 修改 tokenize_batch 函数 ---
    def tokenize_batch(batch):
        source_texts = [item['de'] for item in batch['translation']]
        target_texts = [item['en'] for item in batch['translation']]

        # spacy 的 .pipe() 方法返回的是 Doc 对象
        de_docs = de_tokenizer.pipe(source_texts)
        en_docs = en_tokenizer.pipe(target_texts)

        # 关键修正：
        # 我们在这里直接将 Doc 对象中的每个 Token 对象转换为它的文本字符串。
        # 最终返回的是一个由纯字符串组成的列表的列表。
        batch['source_tokens'] = [[str(token) for token in doc] for doc in de_docs]
        batch['target_tokens'] = [[str(token) for token in doc] for doc in en_docs]

        return batch

    print("开始使用 .map() 高效分词 (这步只在第一次运行时较慢)...")
    tokenized_train_data = train_data.map(
        tokenize_batch,
        batched=True,
        num_proc=16,    # 我的cpu是16核32线程，这里分配16个逻辑单元
        batch_size=1000
    )
    print("分词完成！结果已被缓存。")

    de_token_counter = Counter()
    en_token_counter = Counter()
    max_len_source = 0
    max_len_target = 0

    print("开始从已分词数据中构建词典...")
    # --- 修改这里的循环 ---
    # 因为 item['source_tokens'] 现在已经是字符串列表了，
    # 所以我们不再需要 [str(token) for token in ...] 的转换。
    for item in tqdm(tokenized_train_data):
        source_tokens = item['source_tokens']  # 直接使用
        target_tokens = item['target_tokens']  # 直接使用

        de_token_counter.update(source_tokens)
        en_token_counter.update(target_tokens)

        max_len_source = max(max_len_source, len(source_tokens) + 2)
        max_len_target = max(max_len_target, len(target_tokens) + 2)

    de_vocab = Vocab(de_token_counter, min_freq=2)
    en_vocab = Vocab(en_token_counter, min_freq=2)

    print("\n词典构建与统计完成！")
    print(f"源语言 (de) 词典大小: {len(de_vocab)}")
    print(f"目标语言 (en) 词典大小: {len(en_vocab)}")
    print(f"源语言最长句子长度 (含特殊标记): {max_len_source}")
    print(f"目标语言最长句子长度 (含特殊标记): {max_len_target}")

    return de_vocab, en_vocab, max_len_source, max_len_target


class TranslationDataset(Dataset):
    """
    一个自定义的 PyTorch Dataset，用于翻译任务。
    """
    def __init__(self, data_split, de_vocab, en_vocab, de_tokenizer, en_tokenizer):
        super().__init__()
        self.data = data_split
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 获取原始句子对
        item = self.data[idx]['translation']
        source_text = item['de']
        target_text = item['en']

        # 2. 分词并将 token 转换为 ID
        source_ids = self.de_vocab.text_to_ids(source_text, self.de_tokenizer)
        target_ids = self.en_vocab.text_to_ids(target_text, self.en_tokenizer)

        # 3. 添加句子开始和结束标记
        source_ids = [SOS_IDX] + source_ids + [EOS_IDX]
        target_ids = [SOS_IDX] + target_ids + [EOS_IDX]

        return torch.tensor(source_ids), torch.tensor(target_ids)


def get_dataloaders(config):
    """
    完整的数据加载和预处理流程。

    Args:
        config (dict): 包含所有超参数的配置字典。

    Returns:
        Tuple: (train_dataloader, val_dataloader, de_vocab, en_vocab)
    """
    # 1. 加载原始数据
    raw_datasets = load_raw_data(config['dataset_name'], config['language_pair'])

    # 2. 获取分词器
    en_tokenizer, de_tokenizer = get_tokenizers()

    # 3. 构建词典并获取统计信息
    de_vocab, en_vocab, max_src, max_tgt = build_vocabs_and_get_stats(
        raw_datasets, en_tokenizer, de_tokenizer
    )

    # 4. 为训练集和验证集创建 Dataset 实例
    train_dataset = TranslationDataset(
        raw_datasets['train'], de_vocab, en_vocab, de_tokenizer, en_tokenizer
    )
    val_dataset = TranslationDataset(
        raw_datasets['validation'], de_vocab, en_vocab, de_tokenizer, en_tokenizer
    )

    # 5. 创建 DataLoader 实例
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=generate_batch,
        num_workers = 8,  # <-- 增加！根据你的 CPU 核心数调整，4 或 8 是不错的起点
        pin_memory = True  # <-- 增加！
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # 验证集通常不需要打乱
        collate_fn=generate_batch,
        num_workers = 8,  # <-- 增加！根据你的 CPU 核心数调整，4 或 8 是不错的起点
        pin_memory = True  # <-- 增加！
    )

    return train_dataloader, val_dataloader, de_vocab, en_vocab, max_src, max_tgt


def generate_batch(data_batch):
    """
    自定义的 collate_fn，用于处理一个批次的数据。
    """
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(de_item)
        en_batch.append(en_item)

    # pad_sequence 会自动用 0 来填充，但我们需要用 PAD_IDX
    # batch_first=True 让输出的形状是 (batch_size, seq_len)
    de_batch = pad_sequence(de_batch, batch_first=True, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, batch_first=True, padding_value=PAD_IDX)

    return de_batch, en_batch


# --- 我们可以在这里放一个简单的测试 ---
if __name__ == '__main__':
    # --- 1. 定义一个简单的配置字典用于测试 ---
    # 我们把所有需要的参数都放在这里，模拟从 .yaml 文件加载
    test_config = {
        'dataset_name': 'iwslt2017',
        'language_pair': 'iwslt2017-en-de',
        'batch_size': 4,  # 使用一个小的 batch_size 以方便观察
    }

    # --- 2. 调用我们的主函数 ---
    print("\n--- 开始测试 get_dataloaders 函数 ---")
    try:
        train_loader, val_loader, de_vocab, en_vocab, _, _ = get_dataloaders(test_config)
        print("get_dataloaders 函数成功运行！")
    except Exception as e:
        print(f"get_dataloaders 函数运行时出错: {e}")
        # 如果出错，直接退出
        exit()

    # --- 3. 验证返回的对象类型 ---
    print("\n--- 验证返回对象类型 ---")
    print(f"train_loader 类型: {type(train_loader)}")
    print(f"de_vocab 类型: {type(de_vocab)}")
    assert isinstance(train_loader, DataLoader), "train_loader 不是 DataLoader 对象"
    assert isinstance(de_vocab, Vocab), "de_vocab 不是 Vocab 对象"
    print("对象类型验证通过！")

    # --- 4. 从 train_loader 中取出一个批次并进行检查 ---
    print("\n--- 检查第一个训练批次 ---")
    try:
        # 使用 next(iter(...)) 来安全地获取第一个批次
        first_batch = next(iter(train_loader))
        source_batch, target_batch = first_batch

        print("成功获取第一个批次！")

        # 5. 检查批次的数据类型和形状
        print(f"源批次 (source_batch) 类型: {type(source_batch)}")
        print(f"源批次 (source_batch) 形状: {source_batch.shape}")
        print(f"目标批次 (target_batch) 形状: {target_batch.shape}")

        assert isinstance(source_batch, torch.Tensor), "源批次不是 Tensor"
        assert source_batch.shape[0] == test_config['batch_size'], "源批次的 batch size 不匹配"
        assert target_batch.shape[0] == test_config['batch_size'], "目标批次的 batch size 不匹配"
        print("批次类型和形状验证通过！")

        # 6. 打印批次内容以进行直观检查
        print("\n源批次内容 (Token IDs):")
        print(source_batch)
        print("\n目标批次内容 (Token IDs):")
        print(target_batch)

        # 7. 转换回文本以进行最终验证
        print("\n--- 将第一个批次转换回文本 ---")
        for i in range(test_config['batch_size']):
            source_text = de_vocab.ids_to_text(source_batch[i].tolist())
            target_text = en_vocab.ids_to_text(target_batch[i].tolist())
            print(f"--- 样本 {i + 1} ---")
            print(f"  源: {source_text}")
            print(f"  目标: {target_text}")

    except Exception as e:
        print(f"处理第一个批次时出错: {e}")