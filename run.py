from PIL import Image

from config import config
from helpers import eval
from helpers import parseArguments
from helpers import demo

args = parseArguments.parse('run')
config.setup(args)

# gendage = eval.genderAndAge
newAge = eval.gendageV3


def main():
    """
    demo.process(online=1,
                 labelGenerators=[newAge],
                 inputFile='D:/MsThesis/inputSamples/Recording.mp4',
                 outputFile='output/result.avi',
                 detectionFPS=5,
                 device=config.device)
    """



if __name__ == '__main__':
    main()
