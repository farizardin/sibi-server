cd ..
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_local_bad.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_nonlocal_bad.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_mod_2_bad.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_mod_3_bad.yaml
CUDA_VISIBLE_DEVICES="0" python main_hsd.py --config ./config/hsd/kfold_reduced/joint/full_test_joint_mod_4_bad.yaml