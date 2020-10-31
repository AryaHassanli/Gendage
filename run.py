from config import config
from helpers import eval
from helpers.demo import demo
from helpers.parseArguments import parseArguments

args = parseArguments('main')
if args.gradient:
    config.set(datasetDir='/storage/datasets',
               outputDir='/artifacts')
else:
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir='output')

gendage = eval.genderAndAge


def main():
    demo(online=1,
         labelGenerators=[gendage],
         inputFile='inputSamples/nc.mp4',
         outputFile='output/result.avi',
         detectionFPS=60,
         device=config.device)


if __name__ == '__main__':
    main()
