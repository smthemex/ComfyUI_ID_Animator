# A node using ID_Animator in comfyUI

## NOTICE
You can find ID_Animator in this link  [ID_Animator](https://github.com/ID-Animator/ID-Animator)  

My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     

2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      

3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)

4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       

5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  

6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)

7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)       

8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)

9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)         

10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    

11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)

12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker) 

Update
---
2024-06-15   

1、修复animateddiff帧率上限为32的问题。感谢ShmuelRonen 的提醒   
2、加入face_lora 及lora_adapter的条件控制，模型地址在下面的模型说明里。   
3、加入diffuser 0.28.0以上版本的支持   

1. Fix the issue of animateddiff with a maximum frame rate of 32. Thank you for [ShmuelRonen](https://github.com/ShmuelRonen)
's reminder   
2. Add conditional control for "face_lora" and "lora-adapter", and the model address is provided in the model description below.
3. . Add support for diffuser versions 0.28.0 and above   

--- 既往更新 Previous updates   

1、输出改成单帧图像，方便接其他的视频合成节点，取消原作保存gif动画的选项。  
2、新增模型加载菜单，逻辑上更清晰一些，你可以多放几个动作模型进“.. ComfyUI_ID_Animator/models/animatediff_models”目录   

1. Change the output to a single frame image for easy access to other video synthesis nodes, and remove the option to save the original GIF animation.  
2. Add a new model loading menu to make the logic clearer. You can add a few more action models to the ".. ComfyUI-ID-Animator/models/animateddiff_models" directory

1.Installation  安装   
----
 ``` python 
  git https://github.com/smthemex/ComfyUI_ID_Animator.git
  ```
2  Dependencies  需求库  
-----
If the module is missing, please refer to the separate installation of the missing module in the "if miss module check this requirements.txt" file   

如果缺失模块,请打开"if miss module check this requirements.txt",单独安装缺失的模块


3 Download the checkpoints   下载模型
----

3.1 dir.. ComfyUI_ID_Animator/models  
- Download ID-Animator checkpoint:"animator.ckpt"    [link](https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/)

3.2 dir.. ComfyUI_ID_Animator/models/animatediff_models    
- Download AnimateDiff checkpoint like "/mm_sd_v15_v2.ckpt"    [link](https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/)

3.3 dir.. comfy/models/diffusers  
- Download Stable Diffusion V1.5 all files      [link](https://huggingface.co/spaces/ID-Animator/ID-Animator/tree/main/animatediff/sd)
-  or   
- Download Stable Diffusion V1.5 most files  [link](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) 

3.4 dir.. comfy/models/checkpoints   
- Download "realisticVisionV60B1_v51VAE.safetensors"   [link](https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/)    
  or  any other dreambooth models  
  
3.5 dir.. ComfyUI_ID_Animator/models/image_encoder      
- Download CLIP Image encoder   [link](https://huggingface.co/spaces/ID-Animator/ID-Animator/tree/main/image_encoder)

3.6 dir.. ComfyUI_ID_Animator/models/adapter      
- Download "v3_sd15_adapter.ckpt"   [link](https://huggingface.co/guoyww/animatediff/tree/main)   

3.7  other models       
The first run will download the insightface models to the "X/user/username/.insightface/models/buffalo_l" directory  
   
4 other   其他
----
因为"ID_Animator"作者没有标注开源许可协议，所以我暂时把开源许可协议设置为Apache-2.0 license  
Because "ID_Animator"does not indicate the open source license agreement, I have temporarily set the open source license agreement to Apache-2.0 license   

5 example 示例
----

![](https://github.com/smthemex/ComfyUI_ID_Animator/blob/main/demo/ComfyUI_ID_Animator.gif)



6 Contact "ID_Animator" 
-----
Xuanhua He: hexuanhua@mail.ustc.edu.cn

Quande Liu: qdliu0226@gmail.com

Shengju Qian: thesouthfrog@gmail.com


