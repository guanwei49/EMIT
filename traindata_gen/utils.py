import time
from PIL import Image
import requests
import base64


def encode_image(image_path):
    resolution = 512
    # 打开图像
    with Image.open(image_path) as img:
        width, height = img.size

        # 判断是否需要调整尺寸
        if max(width, height) > resolution:
            if width > height:
                # 宽度是最长边，将其设置为resolution并按比例缩小高度
                new_width = resolution
                new_height = int((new_width / width) * height)
            else:
                # 高度是最长边，将其设置为resolution并按比例缩小宽度
                new_height = resolution
                new_width = int((new_height / height) * width)

            # 调整图像尺寸
            img_resized = img.resize((new_width, new_height))
        else:
            # 原尺寸如果都小于resolution，则不调整
            img_resized = img

        # 临时保存调整后的图像到内存中
        import io
        buffer = io.BytesIO()
        img_resized.save(buffer, format="PNG")  # 您可以选择合适的格式，如PNG、JPEG等
        buffer.seek(0)

        # 进行编码
        return base64.b64encode(buffer.read()).decode('utf-8')

def request(content):
    data = {}

    api_url = 'Your own api_url'
    data['workNo'] = 'Your own workNo'
    data["model"] = "gpt-4-turbo"
    data["messages"] = [{"role": "user", "content": content}]
    data['token'] = 'Your own token'
    headers = {
        "Content-Type": "application/json",
    }

    while True:
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.json()['success']:
                content = response.json()['data']['reply']
                return content
            time.sleep(0.3)
        except requests.exceptions.RequestException as e:
            time.sleep(0.3)