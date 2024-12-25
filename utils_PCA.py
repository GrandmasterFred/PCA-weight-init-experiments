# this will be all for the pca sampling stuff 
# this one checks memory usage 

def check_memory_usage_tensor(target_tensor):
    memory_usage = target_tensor.element_size() * target_tensor.nelement()
    print(f"feature_map memory usage = {memory_usage / (1024 ** 2):.2f} MB")