from src.data.synthetic_position_dataloader import get_dataloaders
from src.models.graphical_model import KalmanGraphicalModel
from src.models.hybrid_inference_model import KalmanHybridInference
from filterpy.kalman import KalmanFilter
from src.utils.data_converters import *
from torch.nn.functional import mse_loss


def evaluate_model_input(model, loader, criterion, device, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]]).t()
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states, inputs) in enumerate(loader):
            obs, states, inputs = obs.to(device), states.to(device), inputs.to(device)

            # compute the prediction.
            if isinstance(model, KalmanHybridInference):
                out, out_list = model(obs, inputs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), inputs.permute(0, 2, 1), 1e-4)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            loss = criterion(out.permute(1, 2, 0), states.permute(2, 1, 0), reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def evaluate_model(model, loader, criterion, device, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]]).t()
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states) in enumerate(loader):
            obs, states = obs.to(device), states.to(device)

            # compute the prediction.
            if isinstance(model, KalmanHybridInference):
                out, out_list = model(obs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), 1e-4)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            # need to override mse averaging.
            loss = criterion(out.permute(1, 2, 0), states.permute(2, 1, 0), reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def evaluate_model_predict(model, loader, criterion, device, n, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]])
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states) in enumerate(loader):
            obs, states = obs.to(device), states.to(device)

            # modify obs to change the last n to 0s
            obs = obs[:, :-n, :]

            # compute the prediction.
            if isinstance(model, KalmanHybridInference):
                out, out_list = model.predict(n, obs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), 1e-4)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            # need to override mse averaging.
            loss = criterion(torch.matmul(H, out.permute(0, 1, 2)), torch.matmul(H, states.permute(0, 2, 1)),
                             reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def compare_models(model_1, model_1_input, model_2, model_2_input, path_to_model1: str,
                   path_to_model2: str):
    # need datasets
    _, _, test_loader = get_dataloaders(train_samples=10,
                                        val_samples=10,
                                        test_samples=20,
                                        sample_length=100,
                                        starting_point=1000,
                                        batch_size=5,
                                        extras=False)
    if model_2_input or model_1_input:
        _, _, test_loader_input = get_dataloaders(train_samples=10,
                                                  val_samples=10,
                                                  test_samples=20,
                                                  sample_length=100,
                                                  starting_point=1000,
                                                  batch_size=5,
                                                  inputs=True,
                                                  extras=False)
    else:
        print("No inputs")
        test_loader_input = test_loader
    # create models
    F = torch.tensor([[1., 1., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1.]])
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]])
    Q = torch.tensor([[0.05 ** 2, 0., 0., 0.],
                      [0., 0.05 ** 2, 0., 0.],
                      [0., 0., 0.05 ** 2, 0.],
                      [0., 0., 0., 0.05 ** 2]])
    R = (0.05 ** 2) * torch.eye(2)
    G = torch.tensor([[1 / 2, 1, 0., 0.],
                      [0., 0., 1 / 2, 1]]).t()
    if model_1_input:
        model1 = model_1(F=F,
                         H=H,
                         Q=Q,
                         R=R,
                         G=G)
    else:
        model1 = model_1(F=F,
                         H=H,
                         Q=Q,
                         R=R)
    if isinstance(model1, KalmanHybridInference):
        model1.gamma = 1e-4
        if len(path_to_model1) > 0:
            model1.load_state_dict(torch.load(path_to_model1, map_location=torch.device("cpu")))

    if model_2_input:
        model2 = model_2(F=F,
                         H=H,
                         Q=Q,
                         R=R,
                         G=G)
    else:
        model2 = model_2(F=F,
                         H=H,
                         Q=Q,
                         R=R)

    if isinstance(model2, KalmanHybridInference):
        model2.gamma = 1e-4

        if len(path_to_model2) > 0:
            print("pretrained")
            model2.load_state_dict(torch.load(path_to_model2, map_location=torch.device("cpu")))
        else:
            print("not pretrained")

    # evaluate and print/return results.
    if model_1_input:
        loss1, av_loss1 = evaluate_model_input(model=model1, loader=test_loader_input,
                                               criterion=mse_loss,
                                               device=torch.device('cpu'))
    else:
        loss1, av_loss1 = evaluate_model(model=model1, loader=test_loader, criterion=mse_loss,
                                         device=torch.device('cpu'))
    if model_2_input:
        loss2, av_loss2 = evaluate_model_input(model=model2, loader=test_loader_input,
                                               criterion=mse_loss,
                                               device=torch.device('cpu'))
    else:
        loss2, av_loss2 = evaluate_model(model=model2, loader=test_loader, criterion=mse_loss,
                                         device=torch.device('cpu'))

    print("Total loss for Model 1: {}".format(loss1))
    print("Average loss for Model 1: {}".format(av_loss1))
    print("Total loss for Model 2: {}".format(loss2))
    print("Total loss for Model 2: {}".format(av_loss2))


def compare2kalman(model, filter, loader):
    """
    Compare HI to classical implementation of Kalman Filter
    :return:
    """
    kal_tot_loss = 0.
    hi_tot_loss = 0.
    for num, (obs, states) in enumerate(loader):
        # set up Kalman Filter things and compute Kalman loss
        x_0 = states[0][0]  # TODO check
        filter.x = torch2numpy(x_0)

        mu, cov, _, _ = filter.batch_filter(torchseq2numpyseq(obs))
        kal_state, P, _, _ = filter.rts_smoother(mu, cov)
        kal_loss = mse_loss(numpy2torch(kal_state).unsqueeze(0), states) #, reduction='sum')

        kal_tot_loss += float(kal_loss)

        # compute HI loss
        hi_state = model.iterate(H.t().matmul(obs.permute(0, 2, 1)), obs, gamma=0.005)
        hi_loss = mse_loss(hi_state.permute(0, 2, 1), states) #, reduction='sum')

        hi_tot_loss += float(hi_loss)

    # print results
    divisor = loader.dataset.total_samples()
    print("Total loss for HI Model: {}".format(hi_tot_loss))
    print("Average loss for HI Model: {}".format(hi_tot_loss / divisor))
    print("Total loss for Kalman Model: {}".format(kal_tot_loss))
    print("Total loss for Kalman Model: {}".format(kal_tot_loss / divisor))


if __name__ == '__main__':
    # compare_models(model_1=KalmanGraphicalModel, model_2=KalmanHybridInference, model_1_input=True,
    #                model_2_input=False,
    #                path_to_model1="",
    #                path_to_model2="")

    # compare_models(model_1=KalmanInputGraphicalModel, model_2=KalmanHybridInference, model_1_input=True,
    #                model_2_input=True,
    #                path_to_model1="",
    #                path_to_model2="../../weighted_mse_results"
    #                               "/weighted_input_train_len5000_mse_start0_seq10.pt")

    F = torch.tensor([[1., 1., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1.]])
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]])
    Q = torch.tensor([[0.15 ** 2, 0., 0., 0.],
                      [0., 0.15 ** 2, 0., 0.],
                      [0., 0., 0.15 ** 2, 0.],
                      [0., 0., 0., 0.15 ** 2]])
    R = (0.5 ** 2) * torch.eye(2)
    G = torch.tensor([[1 / 2, 1, 0., 0.],
                      [0., 0., 1 / 2, 1]]).t()
    P = np.eye(4) * 1000

    hi = KalmanGraphicalModel(F, H, Q, R)

    kal = KalmanFilter(dim_x=4, dim_z=2)
    kal.F = torch2numpy(F)
    kal.H = torch2numpy(H)
    kal.Q = torch2numpy(Q)
    kal.R = torch2numpy(R)
    kal.P = P

    # create test data (from hybrid-inference for comparison)
    # test_state = torch.tensor([[[  711.9885, -2811.0994],
    #      [  713.6472, -2816.0603],
    #      [  715.0543, -2820.4329],
    #      [  716.1395, -2825.0737],
    #      [  717.3056, -2829.3723],
    #      [  718.1702, -2834.0232],
    #      [  719.4956, -2838.5254],
    #      [  720.4882, -2843.2366],
    #      [  721.7919, -2848.0806],
    #      [  723.5571, -2853.1504],
    #      [  725.2791, -2858.3816],
    #      [  726.9940, -2863.2483],
    #      [  728.9388, -2868.3369],
    #      [  731.4630, -2873.7307],
    #      [  733.6902, -2878.8176],
    #      [  735.7360, -2884.3364],
    #      [  738.0704, -2889.8433],
    #      [  740.2575, -2895.6292],
    #      [  742.2587, -2901.2959],
    #      [  744.1218, -2906.9800],
    #      [  745.6492, -2912.9548],
    #      [  747.2455, -2919.0371],
    #      [  748.8893, -2924.9131],
    #      [  750.3260, -2930.7678],
    #      [  751.9581, -2936.3254],
    #      [  753.1091, -2942.2209],
    #      [  754.2488, -2948.0413],
    #      [  755.6708, -2953.8154],
    #      [  756.6486, -2959.6252],
    #      [  757.4707, -2965.1816],
    #      [  758.2794, -2970.7842],
    #      [  758.9647, -2976.2361],
    #      [  760.0148, -2981.6851],
    #      [  761.2555, -2986.6533],
    #      [  762.5586, -2991.7986],
    #      [  763.9714, -2996.9355],
    #      [  765.3505, -3002.1333],
    #      [  766.8204, -3007.7527],
    #      [  768.1000, -3013.3242],
    #      [  769.1920, -3018.6851],
    #      [  770.3719, -3024.0588],
    #      [  771.4703, -3029.4734],
    #      [  772.8389, -3034.8044],
    #      [  773.6382, -3040.4253],
    #      [  774.4871, -3046.0955],
    #      [  774.9698, -3051.9229],
    #      [  775.4612, -3057.4553],
    #      [  776.2851, -3062.9905],
    #      [  777.4609, -3068.4282],
    #      [  778.5762, -3073.9778],
    #      [  780.1403, -3079.6199],
    #      [  781.5546, -3085.0325],
    #      [  782.7630, -3090.4541],
    #      [  783.9415, -3095.4209],
    #      [  785.6682, -3100.5547],
    #      [  787.5818, -3105.9048],
    #      [  789.3712, -3111.5330],
    #      [  791.0800, -3117.2717],
    #      [  792.9969, -3122.9822],
    #      [  794.4144, -3129.0029],
    #      [  795.5262, -3134.8914],
    #      [  796.3383, -3140.4226],
    #      [  797.2274, -3146.3325],
    #      [  798.0857, -3151.7996],
    #      [  799.1606, -3157.4531],
    #      [  800.2087, -3162.9866],
    #      [  801.2933, -3168.5977],
    #      [  802.1639, -3174.2346],
    #      [  803.5392, -3179.9976],
    #      [  804.1484, -3185.7996],
    #      [  804.9079, -3191.7195],
    #      [  805.8298, -3197.7371],
    #      [  806.3856, -3203.8250],
    #      [  806.8882, -3209.7041],
    #      [  807.1158, -3215.4854],
    #      [  807.4520, -3221.3318],
    #      [  807.6339, -3227.6855],
    #      [  808.1352, -3233.4570],
    #      [  808.5810, -3239.4719],
    #      [  808.9980, -3245.2805],
    #      [  809.5392, -3251.5022],
    #      [  810.2609, -3257.4883],
    #      [  811.1933, -3263.4553],
    #      [  812.0132, -3269.5134],
    #      [  812.6121, -3275.8655],
    #      [  813.2197, -3282.2036],
    #      [  814.0742, -3288.0632],
    #      [  814.7692, -3293.8699],
    #      [  815.5650, -3299.4253],
    #      [  816.3067, -3304.9924],
    #      [  816.4932, -3310.7366],
    #      [  816.5368, -3316.3792],
    #      [  816.7480, -3322.0305],
    #      [  816.7817, -3327.5388],
    #      [  817.1888, -3333.2214],
    #      [  817.3610, -3338.8367],
    #      [  817.8713, -3344.4248],
    #      [  817.9315, -3349.6277],
    #      [  818.3701, -3355.0452],
    #      [  818.3757, -3360.7590]]])
    #
    # test_obs = torch.tensor([[[  711.7585, -2810.6897],
    #      [  713.0004, -2815.1995],
    #      [  715.1141, -2820.1794],
    #      [  717.3734, -2824.6799],
    #      [  717.8740, -2828.8809],
    #      [  717.4398, -2834.4438],
    #      [  720.2621, -2838.4822],
    #      [  721.4421, -2842.4404],
    #      [  720.7576, -2847.1687],
    #      [  723.2045, -2853.8162],
    #      [  725.9649, -2858.0249],
    #      [  726.9622, -2863.1765],
    #      [  728.6924, -2868.4731],
    #      [  731.5023, -2874.0889],
    #      [  733.6652, -2878.5325],
    #      [  736.2396, -2884.1838],
    #      [  738.6263, -2890.3772],
    #      [  740.8016, -2895.7729],
    #      [  741.5554, -2901.9592],
    #      [  743.6060, -2907.0439],
    #      [  745.0923, -2912.7217],
    #      [  746.9908, -2919.2026],
    #      [  748.5709, -2924.4912],
    #      [  750.4272, -2931.6765],
    #      [  752.3868, -2935.9700],
    #      [  753.0626, -2942.7683],
    #      [  753.7144, -2946.8560],
    #      [  755.3755, -2953.4092],
    #      [  756.3885, -2960.5022],
    #      [  757.5607, -2965.9285],
    #      [  758.0543, -2970.5439],
    #      [  759.1994, -2976.1418],
    #      [  759.0196, -2981.1345],
    #      [  761.1120, -2986.1135],
    #      [  761.7394, -2992.1047],
    #      [  764.2992, -2996.3020],
    #      [  765.2655, -3001.7751],
    #      [  767.4143, -3007.1475],
    #      [  767.5893, -3013.3730],
    #      [  769.7104, -3018.4758],
    #      [  770.7499, -3023.2063],
    #      [  771.5395, -3029.1323],
    #      [  772.5130, -3035.5986],
    #      [  773.8033, -3040.5496],
    #      [  773.6708, -3046.4531],
    #      [  775.2152, -3051.7214],
    #      [  775.7415, -3057.1338],
    #      [  776.1423, -3063.9167],
    #      [  777.6452, -3067.5537],
    #      [  778.5429, -3073.5161],
    #      [  780.2648, -3080.0090],
    #      [  780.5102, -3084.6860],
    #      [  783.2656, -3090.8220],
    #      [  783.9449, -3095.2957],
    #      [  785.4399, -3101.3428],
    #      [  786.6053, -3106.8984],
    #      [  789.2177, -3110.9104],
    #      [  791.7211, -3118.1094],
    #      [  792.7309, -3123.2939],
    #      [  793.8306, -3128.8901],
    #      [  795.1985, -3135.5610],
    #      [  796.6224, -3140.0234],
    #      [  797.2750, -3146.0393],
    #      [  798.3224, -3151.2815],
    #      [  798.9639, -3157.5479],
    #      [  800.0109, -3163.8198],
    #      [  801.0187, -3167.1543],
    #      [  802.2194, -3174.1389],
    #      [  803.6718, -3179.0967],
    #      [  803.0950, -3186.1597],
    #      [  804.8845, -3191.7058],
    #      [  806.1279, -3197.7642],
    #      [  805.9756, -3203.6489],
    #      [  806.3441, -3209.2710],
    #      [  806.5533, -3215.8708],
    #      [  806.6099, -3221.6277],
    #      [  806.9810, -3226.9978],
    #      [  809.2486, -3234.6724],
    #      [  809.1896, -3239.2104],
    #      [  808.6332, -3244.8738],
    #      [  808.9157, -3252.4548],
    #      [  810.1404, -3257.0237],
    #      [  810.7007, -3263.1152],
    #      [  812.1987, -3268.8445],
    #      [  812.8510, -3276.4666],
    #      [  812.8214, -3281.5884],
    #      [  813.4294, -3287.8828],
    #      [  815.3746, -3294.2766],
    #      [  816.2123, -3299.5942],
    #      [  816.2449, -3303.6331],
    #      [  816.8293, -3310.2317],
    #      [  817.1223, -3317.2014],
    #      [  817.3129, -3322.2146],
    #      [  817.3262, -3328.2087],
    #      [  817.3732, -3333.0779],
    #      [  817.6858, -3338.8760],
    #      [  818.1160, -3344.7563],
    #      [  817.9544, -3349.8367],
    #      [  818.2280, -3354.9507],
    #      [  818.9081, -3360.2351]]])

    # here run through the HI
    # state = hi.iterate(H.t().matmul(test_obs.permute(0, 2, 1)), test_obs, gamma=0.005)
    #
    # # here run through the Kalman Filter
    # x_0 = test_obs[0][0]  # TODO check
    # kal.x = torch2numpy(x_0)
    #
    # mu, cov, _, _ = kal.batch_filter(torchseq2numpyseq(test_obs))
    # kal_state, P, _, _ = kal.rts_smoother(mu, cov)

    _, _, test_loader = get_dataloaders(train_samples=10,
                                        val_samples=10,
                                        test_samples=2,
                                        sample_length=100,
                                        starting_point=0,
                                        batch_size=1,
                                        extras=False)

    # # evaluate_model_predict(hi, test_loader, mse_loss, 'cpu', 2)

    compare2kalman(hi, kal, test_loader)
