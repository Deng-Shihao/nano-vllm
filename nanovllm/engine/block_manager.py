from collections import deque # 用于快速存取空闲的 block ID
import xxhash # 高性能哈希库，这里用于计算 token 序列的唯一标识（便于缓存复用）
import numpy as np # 把 token ID 列表转成字节流，用于哈希

from nanovllm.engine.sequence import Sequence # 之前的类，代表一条推理序列


class Block:

    def __init__(self, block_id):
        self.block_id = block_id # 唯一 ID（在 BlockManager 初始化时分配）
        self.ref_count = 0 # 引用计数（多少序列正在用这个块）
        self.hash = -1 # 块的哈希值（-1 表示未分配或无效）
        self.token_ids = [] # 存放该块的 token ID

    # 写入新的哈希和 token 序列
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 初始化块，引用计数设为 1，内容清空
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size # 每个 block 的大小（通常 256）
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 实际存放所有 block 对象
        self.hash_to_block_id: dict[int, int] = dict() # 哈希到 block 的映射（便于缓存复用）
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲 block ID 列表（FIFO）
        self.used_block_ids: set[int] = set() # 正在使用的 block ID 集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        '''
        function: 给一个 token 序列生成唯一哈希
        prefix: 如果传入 prefix，就把上一个 block 的 hash 也加入计算（链式哈希，防止冲突）

        hash1 = BlockManager.compute_hash(block1)               # 第一个 block 的哈希
        hash2 = BlockManager.compute_hash(block2, prefix=hash1) # 第二个 block 的链式哈希
        '''
        h = xxhash.xxh64()

        if prefix != -1:
            # 加入前一个 block 的哈希值，形成链式哈希
            h.update(prefix.to_bytes(8, "little"))

        # 把 token_ids 转换成字节流后加入哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 从 free_block_ids 拿一个，重置后标记为已用
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # 把引用为 0 的 block 回收到 free_block_ids
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 检查是否有足够的空闲块 供序列使用
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence): # Only once
        '''
        工作流程：
        遍历序列的每个 block。
        计算该块的哈希（最后一块可能不满 → 设为 -1）。
        检查是否已有相同哈希的 block。
        没有 → cache miss → 新分配 block。
        有 → 复用 → 引用计数 +1。
        更新哈希表 hash_to_block_id。
        把分配好的 block_id 存入 seq.block_table。
        '''

        # 一个序列刚分配时，必须还没有 block_table
        assert not seq.block_table

        h = -1 # 上一个 block 的哈希值，初始为 -1
        cache_miss = False # 是否发生 cache miss

        # 遍历当前序列的所有 Block
        for i in range(seq.num_blocks):
            # 取出第 i 个 block 的 token_ids（大小最多 256）
            token_ids = seq.block(i)

            # 如果 block 满（== 256），就用上一个哈希 h 计算链式哈希；
            # 如果不满（说明是最后一个 block），就不用哈希，设为 -1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 查表，看这个哈希是否已经有对应的 block
            block_id = self.hash_to_block_id.get(h, -1)

            # 判断是否 cache miss：
            #   - block_id == -1 → 表示哈希表里没找到
            #   - 或者找到了但内容不一样（哈希冲突 or 上下文不同）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
            # cache miss → 需要新分配 block
                block_id = self.free_block_ids[0] # 取第一个空闲 block_id
                block = self._allocate_block(block_id) # 分配并初始化 block

            else:
            # cache hit → 已经有相同的 block，可以复用
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 如果 block 已经在 used_block_ids，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 否则需要先分配它
                    block = self._allocate_block(block_id)

            # 如果 block 是满的（h != -1），更新哈希和 token_ids
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 记录分配好的 block_id 到序列的 block_table
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        '''
        遍历该序列的所有 block，减少引用计数。
        如果为 0，就释放回收。
        清空 seq.block_table。
        '''
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        '''
        检查：如果追加后要新开一个 block（长度从 block_size 的整数倍 → +1），就需要有空闲块
        '''
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        追加 token 后更新 block 状态。

        参数:
        - seq (Sequence): 需要更新的序列对象

        逻辑:
        1. 如果新 token 是新 block 的第一个 token → 分配新块
        2. 如果 block 正好填满 → 计算哈希并更新映射
        3. 如果在 block 中间 → 保持 hash = -1
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        # 情况1: 新 token 是新 block 的第一个
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        # 情况2: block 填满时
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        # 情况3: block 尚未填满
        else:
            assert last_block.hash == -1
