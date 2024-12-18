
# TODO:
将attention算法部署到vLLM中attention后端中：
ring_flash_attn: https://github.com/zhuzilin/ring-flash-attention
chunk_attn: https://github.com/microsoft/chunk-attention
# 1. downdload vllm repo
git clone git@github.com:vllm-project/vllm.git
# 2. 修改vLLM代码及重新编译
参考install.sh，拷贝install.sh到vllm的根目录执行即可，注意安装的版本问题
# 3. 部署计算到vLLM中
attention算法写入到vllm/attention/backends内，实现参考flash_attn.py
