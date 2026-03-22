import os
import shutil

def main():
    # 定义文件夹路径
    folder = r'd:\code\HiDDeN-master\train2017'
    train_folder = r'd:\code\HiDDeN-master\train'
    
    # 确保train文件夹存在
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        print(f"创建train文件夹: {train_folder}")
    
    # 获取train2017中所有jpg图片
    images = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    
    # 按文件名排序
    images.sort()
    
    total_images = len(images)
    move_count = 5000
    
    if total_images < move_count:
        print(f"警告: train2017只有{total_images}张图片，少于需要移动的{move_count}张")
        move_count = total_images
    
    # 获取需要移动的图片（前5000张）
    images_to_move = images[:move_count]
    
    print(f"train2017共有{total_images}张图片")
    print(f"将移动前{move_count}张图片到train文件夹...")
    
    # 移动图片
    moved_count = 0
    for img_name in images_to_move:
        src_path = os.path.join(folder, img_name)
        dst_path = os.path.join(train_folder, img_name)
        
        try:
            shutil.move(src_path, dst_path)
            moved_count += 1
            if moved_count % 500 == 0:
                print(f"已移动 {moved_count}/{move_count} 张图片")
        except Exception as e:
            print(f"移动 {img_name} 失败: {e}")
    
    print(f"\n完成! 共移动了 {moved_count} 张图片到train文件夹")
    
    # 验证结果
    remaining = len([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])
    train_count = len([f for f in os.listdir(train_folder) if f.lower().endswith('.jpg')])
    print(f"\n验证结果:")
    print(f"train2017剩余图片: {remaining} 张")
    print(f"train文件夹现有图片: {train_count} 张")

if __name__ == "__main__":
    main()
