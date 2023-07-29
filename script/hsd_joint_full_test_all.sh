cd ..
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_local.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_nonlocal.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_mod_2.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_mod_3.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_mod_4.yaml