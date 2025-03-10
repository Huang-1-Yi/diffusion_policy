# CQU_KS 2024-7-23
# 场景理解Agent

print('导入视觉大模型模块')
import cv2
import numpy as np
import time
from PIL import Image
from PIL import ImageFont, ImageDraw

# 导入中文字体，指定字号
font = ImageFont.truetype('/home/ks/vlm-robot/asset/SimHei.ttf', 26)
# 系统提示词
SYSTEM_PROMPT = '''
我即将给你一句话，介绍现在的场景，你帮我理解目标物体可能在的位置，理解一下场景中的可能藏有物体的容器

例如，如果我的指令是：现在场景中有瓶子、罐子、牙签盒、冰箱、抽屉、架子，我现在想找一个苹果但是它不在桌面，他可能在哪些物品里

你输出应该为：

冰箱、抽屉、架子

例如，如果我的指令是：现在场景中有筷子、碗、盆、水瓶、菜板，我现在想找一个盘子但是它不在桌面，他可能在哪些物品里

你输出应该为：

None

我现在的指令是：
'''
# Yi-Vision调用函数
import openai
from openai import OpenAI
import base64
def yi_vision_api(PROMPT):
    '''
    零一万物大模型开放平台，yi-vision视觉语言多模态大模型API
    '''

    client = OpenAI(
        api_key='1e833b4730074ee8abb11aa9c5d022b3',
        base_url="https://api.lingyiwanwu.com/v1"
    )

    # 向大模型发起请求
    completion = client.chat.completions.create(
        model="yi-vision",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT+PROMPT
                    },

                ]
            },
        ]
    )

    # 解析大模型返回结果
    result = completion.choices[0].message.content.strip()
    print('    大模型调用成功！')
    return result

def sence_agent(objects_dic,obj="瓶子"):

    keys_string = ','.join(objects_dic.keys())
    prompt=SYSTEM_PROMPT +"现在场景中有"+keys_string+"我现在的指令是找到"+obj+"，他可能在哪些物品里面？你只可以回复物体名称不可回复其他的中间用“、”隔开，如果没有就回复None"
    print(prompt)
    result=yi_vision_api(prompt)
    if result=='没有':
        print("场景中没有目标物体")
        return [result]
    else:
        parts = result.split("、")
        return parts

if __name__=='__main__':

    objects= {"篮子1": [90,80,305,609],"篮子2": [300,203,507,509],"红牛": [90,80,305,609],"饼干": [300,203,507,509],"牙膏": [90,80,305,609],"牙刷": [300,203,507,509]}
    print(sence_agent(objects,"遥控器"))
