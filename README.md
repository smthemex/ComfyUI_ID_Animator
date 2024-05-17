# A node using ID_Animator in comfyUI

## NOTICE
You can find ID_Animator in this link  [ID_Animator](https://github.com/ID-Animator/ID-Animator)  

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
- Download ID-Animator checkpoint https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/animator.ckpt  

3.2 dir.. ComfyUI_ID_Animator/models/animatediff_models    
- Download AnimateDiff checkpoint https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/mm_sd_v15_v2.ckpt   

3.3 dir.. comfy/models/diffusers  
- Download Stable Diffusion V1.5 all files  https://huggingface.co/spaces/ID-Animator/ID-Animator/tree/main/animatediff/sd   
  or   
- Download Stable Diffusion V1.5 most files https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main      

3.4 dir.. comfy/models/checkpoints   
- Download realisticVisionV60B1 https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/realisticVisionV60B1_v51VAE.safetensors  
  or  any other dreambooth models  
  
3.5 dir.. ComfyUI_ID_Animator/models/image_encoder      
- Download CLIP Image encoder https://huggingface.co/spaces/ID-Animator/ID-Animator/tree/main/image_encoder   

3.6  other models       
The first run will download the insightface models to the "X/user/username/.insightface/models/buffalo_l" directory  
   
4 other   其他
----
因为"ID_Animator"作者没有标注开源许可协议，所以我暂时把开源许可协议设置为Apache-2.0 license  
Because "ID_Animator"does not indicate the open source license agreement, I have temporarily set the open source license agreement to Apache-2.0 license   

5 example 示例
----

![](https://github.com/smthemex/ComfyUI_ID_Animator/blob/main/demo/example.png)


6 Contact "ID_Animator" 
-----
Xuanhua He: hexuanhua@mail.ustc.edu.cn

Quande Liu: qdliu0226@gmail.com

Shengju Qian: thesouthfrog@gmail.com


