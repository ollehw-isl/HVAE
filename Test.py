# %%
########################################################################
# import default python-library
########################################################################
import os
import torch
########################################################################


########################################################################
# import additional python-library
########################################################################
import pandas
import numpy
from tqdm import tqdm

# original lib
import common as com
import Model as HVAE
########################################################################
from torch.autograd import Variable

########################################################################

n_mels = 128
frames = 5
n_fft = 2048
hop_length = 1024
power = 2.0

# %% Hyper parameter
# nsamples: 하나의 예측에 마스크 업데이트해서 이들의 평균 예측으로
mode = True
Setting_num = '12'
model_load_dir = 'model/Setting_' +  Setting_num + '.pt'
target_dir = "/nas001/users/mg.choi/work/sound.ai/sound.ai.machine/qbank_comp/repro_tdms/train_9class_new2"
os.makedirs('./result', exist_ok=True)
anomaly_score_csv1 = "result/Setting_" +  Setting_num + "_score_OK.csv"
anomaly_score_csv2 = "result/Setting_" +  Setting_num + "_score_OK_unseen.csv"
anomaly_score_csv3 = "result/Setting_" +  Setting_num + "_score_NG.csv"


BN = True
embedding_dim = 16
layers = [640, 640, 256, 128]
Beta = 2


# %%   
# nsamples: 하나의 예측에 마스크 업데이트해서 이들의 평균 예측으로

if __name__ == "__main__":
    OK_csv = pandas.read_csv('/home/jongwook95.lee/Audio_anomal/Comp/Data/both_vib_noise.csv', header=None)
    files = list(OK_csv.sample(n=4000, random_state=4917)[0])

    print("============== MODEL LOAD ==============")
    # set model path
    
    # load model file
    device = torch.device('cuda')
    if BN == True:
        model = HVAE.VAE_BN(layers=layers, embedding_dim=embedding_dim)
    else:
        model = HVAE.VAE_NoBN(layers=layers, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_load_dir))
    model.to(device)
    model.eval()


    file_name_list = []
    anomaly_score_list = []
    print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
    with torch.no_grad():
        for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
            file_name_list = file_name_list + [file_path.split('/')[-1]]
        
            test_data = com.tdms_file_to_vector_array(file_path, channel = 'timesignal_v1',
                                                n_mels = n_mels,
                                                frames = frames,
                                                n_fft = n_fft,
                                                hop_length = hop_length,
                                                power = power)

            test_data = torch.from_numpy(test_data).cuda()
            test_data = Variable(test_data).float()
            recon_x, recon_sigma, mu, logvar = model(test_data)
            loss = HVAE.loss_function_test(Beta, recon_x, recon_sigma, test_data, mu, logvar)

            loss_average = torch.mean(loss).cpu().detach()
            loss_median = torch.median(loss).cpu().detach()
            loss_max = torch.max(loss).cpu().detach()
            loss_quantile = torch.quantile(loss, 0.25).cpu().detach()
            loss_top100 = list(torch.topk(loss, 30).values.cpu().detach())
            loss_list = [loss_average, loss_median, loss_max, loss_quantile] + loss_top100
            loss = numpy.array([loss_list])
            
            # anomaly_score_list = anomaly_score_list + [torch.max(torch.topk(loss, K).values)]
            anomaly_score_list = anomaly_score_list + [loss]

        # save anomaly score
    
    anomaly_score_list = numpy.vstack(anomaly_score_list)
    result1 = pandas.DataFrame(anomaly_score_list)
    result = pandas.DataFrame({'File': file_name_list})
    result = pandas.concat([result, result1], axis = 1)
    result.to_csv(anomaly_score_csv1, index = False)
    com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv2))

# nsamples: 하나의 예측에 마스크 업데이트해서 이들의 평균 예측으로

if __name__ == "__main__":
    OK_csv = pandas.read_csv('/home/jongwook95.lee/Audio_anomal/Comp/Data/both_vib_noise.csv', header=None)
    files = list(OK_csv.sample(n=2000, random_state=8312)[0])

    print("============== MODEL LOAD ==============")
    # set model path
    
    # load model file
    device = torch.device('cuda')
    if BN == True:
        model = HVAE.VAE_BN(layers=layers, embedding_dim=embedding_dim)
    else:
        model = HVAE.VAE_NoBN(layers=layers, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_load_dir))
    model.to(device)
    model.eval()


    file_name_list = []
    anomaly_score_list = []
    print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
    with torch.no_grad():
        for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
            file_name_list = file_name_list + [file_path.split('/')[-1]]
        
            test_data = com.tdms_file_to_vector_array(file_path, channel = 'timesignal_v1',
                                                n_mels = n_mels,
                                                frames = frames,
                                                n_fft = n_fft,
                                                hop_length = hop_length,
                                                power = power)

            test_data = torch.from_numpy(test_data).cuda()
            test_data = Variable(test_data).float()
            recon_x, recon_sigma, mu, logvar = model(test_data)
            loss = HVAE.loss_function_test(Beta, recon_x, recon_sigma, test_data, mu, logvar)

            loss_average = torch.mean(loss).cpu().detach()
            loss_median = torch.median(loss).cpu().detach()
            loss_max = torch.max(loss).cpu().detach()
            loss_quantile = torch.quantile(loss, 0.25).cpu().detach()
            loss_top100 = list(torch.topk(loss, 30).values.cpu().detach())
            loss_list = [loss_average, loss_median, loss_max, loss_quantile] + loss_top100
            loss = numpy.array([loss_list])
            
            # anomaly_score_list = anomaly_score_list + [torch.max(torch.topk(loss, K).values)]
            anomaly_score_list = anomaly_score_list + [loss]

        # save anomaly score
    
    anomaly_score_list = numpy.vstack(anomaly_score_list)
    result1 = pandas.DataFrame(anomaly_score_list)
    result = pandas.DataFrame({'File': file_name_list})
    result = pandas.concat([result, result1], axis = 1)
    result.to_csv(anomaly_score_csv2, index = False)
    com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv2))


# %%
if __name__ == "__main__":
    NG_csv = pandas.read_csv('/home/jongwook95.lee/Audio_anomal/Comp/Data/real_ngs.csv', header=None)
    files = list(NG_csv[0])

    print("============== MODEL LOAD ==============")
    # set model path
    
    # load model file
    device = torch.device('cuda')
    if BN == True:
        model = HVAE.VAE_BN(layers=layers, embedding_dim=embedding_dim)
    else:
        model = HVAE.VAE_NoBN(layers=layers, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_load_dir))
    model.to(device)
    model.eval()


    file_name_list = []
    anomaly_score_list = []
    print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
    with torch.no_grad():
        for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
            file_name_list = file_name_list + [file_path.split('/')[-1]]
        
            test_data = com.tdms_file_to_vector_array(file_path, channel = 'timesignal_v1',
                                                n_mels = n_mels,
                                                frames = frames,
                                                n_fft = n_fft,
                                                hop_length = hop_length,
                                                power = power)

            test_data = torch.from_numpy(test_data).cuda()
            test_data = Variable(test_data).float()
            recon_x, recon_sigma, mu, logvar = model(test_data)
            loss = HVAE.loss_function_test(Beta, recon_x, recon_sigma, test_data, mu, logvar)

            loss_average = torch.mean(loss).cpu().detach()
            loss_median = torch.median(loss).cpu().detach()
            loss_max = torch.max(loss).cpu().detach()
            loss_quantile = torch.quantile(loss, 0.25).cpu().detach()
            loss_top100 = list(torch.topk(loss, 30).values.cpu().detach())
            loss_list = [loss_average, loss_median, loss_max, loss_quantile] + loss_top100
            loss = numpy.array([loss_list])
            
            # anomaly_score_list = anomaly_score_list + [torch.max(torch.topk(loss, K).values)]
            anomaly_score_list = anomaly_score_list + [loss]

        # save anomaly score
    
    anomaly_score_list = numpy.vstack(anomaly_score_list)
    result1 = pandas.DataFrame(anomaly_score_list)
    result = pandas.DataFrame({'File': file_name_list})
    result = pandas.concat([result, result1], axis = 1)
    result.to_csv(anomaly_score_csv3, index = False)
    com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv2))

# %%
