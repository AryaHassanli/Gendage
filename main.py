from config import config
from helpers import parseArguments

# Handle arguments
args = parseArguments.parse('main')
if args.gradient:
    config.set(datasetDir='/storage/datasets',
               outputDir='/artifacts')
else:
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir='output')


def main():
    pass


if __name__ == '__main__':
    main()
