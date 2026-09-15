import gdown
# 使用列表组织路径与 URL（防止 key 重复被覆盖）
links = [# 保存目录默认是文件名，如果要放到某个目录下，保存目录是目录/
    # (
    #     'data/waymo/',
    #     'https://drive.google.com/uc?id=1grtLyo6zQ12-rztdUXQWZOneuJ8tnQd_'
    # ),
    (
        'video_diffusion/ckpts/',  
        'https://drive.google.com/file/d/1Qtdkm0wvIUSMWQMVldd-d16rHZsNFFt1/view?usp=drive_link',
    ),
    (
        'video_diffusion/ckpts/',  
        'https://drive.google.com/file/d/1bCM7XLDquRqnnpauQAK5j1jP-n0y1ama/view?usp=drive_link',
    ),
]
for output_path, link in links:
    gdown.download(link, output_path, quiet=False,fuzzy=True)# 下载文件