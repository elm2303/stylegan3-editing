from pathlib import Path

dataset_paths = {
	'celeba_train': Path(''),
	'celeba_test': Path(''),

	'ffhq': Path(''),
	'ffhq_unaligned': Path('')
}

model_paths = {
	# models for backbones and losses
	'ir_se50': Path('pretrained_models/model_ir_se50.pth'),
	# stylegan3 generators
	'stylegan3_ffhq': Path('pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'),
	'stylegan3_ffhq_pt': Path('pretrained_models/stylegan3-r-ffhq-1024x1024.pt'),
	'stylegan3_ffhq_unaligned': Path('pretrained_models/stylegan3-r-ffhqu-1024x1024.pkl'),
	'stylegan3_ffhq_unaligned_pt': Path('pretrained_models/sg3-r-ffhqu-1024.pt'),
	# model for face alignment
	'shape_predictor': Path('pretrained_models/shape_predictor_68_face_landmarks.dat'),
	# models for ID similarity computation
	'curricular_face': Path('pretrained_models/CurricularFace_Backbone.pth'),
	'mtcnn_pnet': Path('pretrained_models/mtcnn/pnet.npy'),
	'mtcnn_rnet': Path('pretrained_models/mtcnn/rnet.npy'),
	'mtcnn_onet': Path('pretrained_models/mtcnn/onet.npy'),
	# classifiers used for interfacegan training
	'age_estimator': Path('pretrained_models/dex_age_classifier.pth'),
	'pose_estimator': Path('pretrained_models/hopenet_robust_alpha1.pkl')
}

styleclip_directions = {
	"ffhq": {
		'delta_i_c': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/delta_i_c.npy'),
		's_statistics': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/s_stats'),
	},
	'templates': Path('editing/styleclip_global_directions/templates.txt')
}

interfacegan_aligned_edit_paths = {
	'pose': Path('editing/interfacegan/boundaries/pose_boundary.npy')
 }
