from torch.utils.tensorboard import SummaryWriter
import zipfile
from utilities.util_str2url import getTBurl

def setupWriter(args, save_folder, params, net, graphinput,YMLFILENAME):
    writer = SummaryWriter(log_dir=save_folder)
    writer.add_text('arguments',str(vars(args)),0)
    writer.flush()
    writer.add_scalar('params/', params)
    writer.add_graph(net, graphinput)
    # Store files
    zf = zipfile.ZipFile(save_folder + "/files.zip", "w")
    for ff in args.files_to_store:
        zf.write(ff)
    zf.write(YMLFILENAME)
    zf.close()
    print("# Files stored in ", save_folder + "/files.zip")
    genShowTB(save_folder,"showTB.bat")
    return writer

def genShowTB(save_folder,filename):
    with open(save_folder + "/" + filename, "w") as f:
        f.write('start powershell -command "tensorboard --logdir=.\ --port=6006"\n')
        f.write('timeout 2\n')
        f.write('start '+getTBurl("localhost")+'\n')

if __name__ == "__main__":
    genShowTB("C:/Users/xhua37/Desktop/","test.bat")