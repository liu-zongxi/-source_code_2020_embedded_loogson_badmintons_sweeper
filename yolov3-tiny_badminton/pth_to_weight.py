from models import *
model = Darknet("config/yolov3-tiny.cfg")
model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_99.pth", map_location=torch.device("cuda")))
model.save_darknet_weights("weights/slected_ckpt.weights")