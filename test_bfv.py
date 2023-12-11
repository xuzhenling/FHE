import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import tenseal as ts
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 初始化 BFV 上下文
ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=65537)
ctx.global_scale = pow(2, 21)

# 生成密钥
sk = ctx.secret_key()

# 向上下文添加公共密钥
ctx.make_context_public()

# 加载并加密图像
image_folder = "original_images"
encrypted_images = []

encryption_times = []
decryption_times = []

# 创建保存解密图像的文件夹
decrypted_folder = "decrypted_images"
os.makedirs(decrypted_folder, exist_ok=True)

# 获取原始图像的形状
original_shape = (64, 64)

ssim_values = []  # 存储所有SSIM值

for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)

        # 加载图像
        original_image = Image.open(image_path)

        # 将图像数据转换为 NumPy 数组
        original_array = np.array(original_image)

        # 确保图像大小为 64x64
        original_array = original_array[:64, :64]

        # 加密图像
        start_time = time.time()
        encrypted_image = ts.bfv_vector(ctx, original_array.flatten().tolist())
        encryption_time = time.time() - start_time
        encrypted_images.append(encrypted_image)
        
        encryption_times.append(encryption_time)

        print(f"图像已加密: {filename}, 加密时间: {encryption_time:.4f} 秒")

        # 解密图像
        start_time = time.time()
        decrypted_filename = f"decrypted_{filename}"
        decrypted_image = np.array(encrypted_image.decrypt(sk)).reshape(original_shape)
        decryption_time = time.time() - start_time
        decryption_times.append(decryption_time)
        print(f"图像已解密: {filename}, 解密时间: {decryption_time:.4f} 秒")
        
        # 转换为 NumPy 数组
        decrypted_array = np.array(decrypted_image)
        
        # 假设 image_array 是你的图像数组
        min_value = np.min(original_array)
        max_value = np.max(decrypted_array)
        
        # 计算 data_range
        data_range = max_value - min_value
        ssim_value, _ = ssim(original_array, decrypted_array, data_range=data_range, full=True)
        ssim_values.append(ssim_value)  # 存储SSIM值
        print(f"SSIM值: {ssim_value:.16f}")
        
        # 保存图像
        image = Image.fromarray(decrypted_array.astype(np.uint8))
        image.save(os.path.join(decrypted_folder, decrypted_filename))

# 计算加密时间的平均值
avg_encryption_time = sum(encryption_times) / len(encryption_times)
print(f"平均加密时间: {avg_encryption_time:.4f} 秒")

# 计算解密时间的平均值
avg_decryption_time = sum(decryption_times) / len(decryption_times)
print(f"平均解密时间: {avg_decryption_time:.4f} 秒")

# 计算所有SSIM值的平均值
avg_ssim_value = sum(ssim_values) / len(ssim_values)
print(f"平均SSIM值: {avg_ssim_value:.16f}")

# 执行同态加法、同态缩放等操作
result_image = sum(encrypted_images)  # 例子：同态加法

# 解密结果
start_time = time.time()
decrypted_image = np.array(result_image.decrypt(sk)).reshape(original_shape)
decryption_time = time.time() - start_time
print(f"解密时间: {decryption_time:.4f} 秒")

# 加载并加密查询图像
query_image_path = "match_image.png"
query_image = Image.open(query_image_path).convert("L")
query_array = np.array(query_image)
query_array = query_array[:64, :64]
encrypted_query_image = ts.bfv_vector(ctx, query_array.flatten().tolist())


# 使用加密的数据库图像计算相似性分数
similarity_scores = []
for encrypted_image in encrypted_images:
    decrypted_array = np.array(encrypted_image.decrypt(sk)).reshape(original_shape)
    ssim_value, _ = ssim(query_array, decrypted_array, data_range=255.0, full=True)
    similarity_scores.append(ssim_value)

# 找到最佳匹配的索引
best_match_index = np.argmax(similarity_scores)
best_match_filename = os.listdir(image_folder)[best_match_index]

# 显示最佳匹配
print(f"查询图像的最佳匹配: {best_match_filename}, SSIM: {similarity_scores[best_match_index]}")

print("完成")
