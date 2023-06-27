import os

folder_A = "文件夹A的路径"
folder_B = "文件夹B的路径"

# 获取文件夹A中的图片文件名（去除后缀）
image_files = [os.path.splitext(file)[0] for file in os.listdir(folder_A) if file.endswith((".jpg", ".png"))]

# 获取文件夹B中的txt文件名（去除后缀）
txt_files = [os.path.splitext(file)[0] for file in os.listdir(folder_B) if file.endswith(".txt")]

# 找到在文件夹A中存在但在文件夹B中没有对应txt文件的图片文件名
files_to_delete = set(image_files) - set(txt_files)

# 删除这些图片文件
for file in files_to_delete:
    file_path = os.path.join(folder_A, file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"已删除文件：{file_path}")

print("完成删除操作。")
