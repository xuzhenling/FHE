import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import tenseal as ts
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim


# 初始化 CKKS 上下文
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
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

# 创建保存加密和解密图像的文件夹
decrypted_folder = "decrypted_images"
os.makedirs(decrypted_folder, exist_ok=True)

# 获取原始图像的形状
original_shape = (64, 64)

ssim_values = []  # 用于保存每对图像的 SSIM 值

for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)

        # 加载图像
        original_image = Image.open(image_path)

        # 将图像数据转换为 NumPy 数组
        original_array = np.array(original_image).astype(float)

        # 确保图像大小为 64x64
        original_array = original_array[:64, :64]

        # 加密图像
        start_time = time.time()
        encrypted_image = ts.ckks_vector(ctx, original_array.flatten().tolist())
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
        decrypted_array = np.array(decrypted_image).astype(float)

        # 计算 SSIM 值
        ssim_value, _ = ssim(original_array, decrypted_array, data_range=255.0, full=True)
        ssim_values.append(ssim_value)

        print(f"SSIM值: {ssim_value}")

        # 保存图像
        image = Image.fromarray(decrypted_array.astype(np.uint8))
        image.save(os.path.join(decrypted_folder, decrypted_filename))

# 计算加密时间的平均值
avg_encryption_time = sum(encryption_times) / len(encryption_times)
print(f"平均加密时间: {avg_encryption_time:.4f} 秒")

# 计算解密时间的平均值
avg_decryption_time = sum(decryption_times) / len(decryption_times)
print(f"平均解密时间: {avg_decryption_time:.4f} 秒")

# 计算平均 SSIM 值
avg_ssim = sum(ssim_values) / len(ssim_values)
print(f"平均 SSIM 值: {avg_ssim}")

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
query_array = np.array(query_image).astype(float)
query_array = query_array[:64, :64]
encrypted_query_image = ts.ckks_vector(ctx, query_array.flatten().tolist())

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
print(f"Best match for the query image: {best_match_filename}, SSIM: {similarity_scores[best_match_index]}")

print("完成")
