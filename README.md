# -ResNet-
一个基于ResNet模型的人脸识别项目
-	使用Dlib的ResNet模型提取已知人脸图像特征向量A
-	使用OpenCV控制笔记本摄像头拍摄人脸图像作为待检测人脸图像
-	提取待检测人脸图像特征向量B，并与向量A计算相似度
-	将识别结果在屏幕上进行输出（相似度大于阈值的为同一人）