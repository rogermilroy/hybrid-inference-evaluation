import torch
from torch.jit import ScriptModule

if __name__ == '__main__':
    # y = torch.load("../../../catkin_ws/recording/odom-0.pt")
    x = torch.jit.load("../../../catkin_ws/recording/y-0.pt")
    print(x)
    print(next(x.named_parameters()))

    # print(y)

    # # THIS IS TO REMOVE UNNEEDED SAMPLES
    #
    # i = 0
    #
    # lowest_diff = 10000000.  # some high number
    #
    # for j in range(100):   # this is the number of odom readings.
    #     # compare timestamps of odom and ys. i to j
    #     # load the files and get the time stamps
    #     try:
    #         x: ScriptModule = torch.jit.load("../../../catkin_ws/recording/y-" + str(i) + ".pt")
    #         y = torch.load("../../../catkin_ws/recording/odom-"+str(j)+".pt")
    #     except:
    #         print("Unable to load, probably ran out.")
    #         break
    #     _, t1 = next(x.named_parameters())
    #     t2 = y[0]
    #     diff = abs(t1 - t2)
    #     pass
    #     # if lower than current diff
    #     if diff < lowest_diff:
    #         # then update lower diff
    #         lowest_diff = diff
    #         # drop file with j-1
    #         if j > 0:
    #             os.remove("../../../catkin_ws/recording/odom-"+str(j-1)+".pt")
    #             print('removing {}'.format(j-1))
    #         # go to next iteration??
    #     # if higher or equal than current diff
    #     else:
    #         # increment i (move to the next ys) but leave j (the file remains)
    #         i += 1
    #         # reset diff to high number.
    #         lowest_diff = 1000000000.
    #
    # # THIS IS TO RENUMBER SO THAT THE SAMPLES LINE UP PROPERLY
    #
    # k = 0
    # for i in range(100):  # this should be the number of odoms before
    #     if os.path.exists("../../../catkin_ws/recording/odom-"+str(i)+".pt"):
    #         os.rename("../../../catkin_ws/recording/odom-"+str(i)+".pt", "../../../catkin_ws/recording/odom-"+str(k)+".pt")
    #         print(torch.load("../../../catkin_ws/recording/odom-" + str(k) + ".pt"))
    #         k += 1
