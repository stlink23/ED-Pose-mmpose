# ED-Pose-mmpose
suport train EDPose in mmpose

1.第一步：把文件放在对应的文件夹里
mmpose\models\heads\transformer_heads\edpose_head.py
mmpose\models\heads\transformer_heads\criterion.py
mmpose\models\heads\transformer_heads\matcher.py
mmpose\models\heads\transformer_heads\transformers\utils.py
E:\code\mmpose\mmpose\models\heads\transformer_heads\transformers\box_ops.py
2第二步，运行
修改train.py里面的参数
python mmpose\tools\train.py

参考:https://github.com/IDEA-Research/ED-Pose
