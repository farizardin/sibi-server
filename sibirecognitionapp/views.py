from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from django.http import JsonResponse
import json
# from django import forms
from django.core.files.storage import FileSystemStorage
from eval import Eval
from forms.file_form import FileForm
import time
import os
# Create your views here.
class SibiRecognition(APIView):
    def post(self, request, *args, **kwargs):
        try:
            if request.method == 'POST':
                if request.FILES.get("video") is None:
                    error = {"error": {"message": "Atribut 'video' diperlukan"}, "code": 422}
                    return JsonResponse(error, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
                form = FileForm(request.POST, request.FILES)
                if not form.is_valid():
                    error = {"error": {"message": "Berkas tidak didukung. Berkas yang didukung adalah video bertipe MP4"}, "code": 422}
                    return JsonResponse(error, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

                path = "./videos"
                myfile = request.FILES['video']
                fs = FileSystemStorage()
                save_path = os.path.join(path, myfile.name)
                filename = fs.save(save_path, myfile)
                uploaded_file_url = '.' + fs.url(filename)
                config_path = './config/recalculated/test_sibi_with_mouth_augmented_individual_outer_group_shift_joint_mod_3.yaml'
                predict = Eval(config_path, uploaded_file_url)
                predict.keypoints_normalization_method = 'recalculate_coordinates'
                start_eval_process = time.time()
                data = predict.eval()
                finish_eval_process = time.time()
                total_process_elapsed = finish_eval_process - start_eval_process
                data["total_process_elapsed_time"] = round(total_process_elapsed, 2)
                result = {"data": data}
                return JsonResponse(result, status=status.HTTP_200_OK)
        except Exception as err:
            error = {"error": {"message": f"Terjadi kesalahan pada server. {err}"}, "code": 500}
            return JsonResponse(error, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
