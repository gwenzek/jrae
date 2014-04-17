package main;

import io.DataSet;
import io.LabeledDataSet;
import io.MatProcessData;
import io.ParsedReviewData;

import java.io.*;
import java.util.Map;

import parallel.Parallel;
import parallel.ThreadPool;

import util.CommandLineUtils;
import classify.LabeledDatum;

public class Arguments {

    String dir = "data/parsed/";
    int minCount = DataSet.MINCOUNT;
    boolean isTestLabelsKnown = false;

    String ModelFile = null;
    String treeDumpDir = null;
    String WordMapFile = null;
    String LabelMapFile = null;
    String ClassifierFile = null;
    String featuresOutputFile = null;
    String ProbabilitiesOutputFile = null;

    String matrixDumpDir = null;

    boolean trainModel = false;
    int numFolds = 10, maxIterations = 80, embeddingSize = 50, catSize = 1;
    int dictionarySize, hiddenSize, visibleSize;
    double AlphaCat = 0.2, Beta = 0.5;
    double[] Lambda = new double[]{1e-05, 0.0001, 1e-05, 0.01};

    boolean exitOnReturn = false;

    LabeledDataSet<LabeledDatum<Integer, Integer>, Integer, Integer> Dataset = null;

    public void parseArguments(String[] args) throws IOException {
        Map<String, String> argMap = CommandLineUtils
                .simpleCommandLineParser(args);

        if (argMap.containsKey("-minCount"))
            minCount = Integer.parseInt(argMap.get("-minCount"));

        if (argMap.containsKey("-numFolds"))
            numFolds = Integer.parseInt(argMap.get("-numFolds")) - 1;

        if (argMap.containsKey("-maxIterations"))
            maxIterations = Integer.parseInt(argMap.get("-maxIterations"));

        if (argMap.containsKey("-embeddingSize"))
            embeddingSize = Integer.parseInt(argMap.get("-embeddingSize"));

        if (argMap.containsKey("-alphaCat"))
            AlphaCat = Double.parseDouble(argMap.get("-alphaCat"));

        if (argMap.containsKey("-lambdaW"))
            Lambda[0] = Double.parseDouble(argMap.get("-lambdaW"));

        if (argMap.containsKey("-lambdaL"))
            Lambda[1] = Double.parseDouble(argMap.get("-lambdaL"));

        if (argMap.containsKey("-lambdaCat"))
            Lambda[2] = Double.parseDouble(argMap.get("-lambdaCat"));

        if (argMap.containsKey("-lambdaLRAE"))
            Lambda[3] = Double.parseDouble(argMap.get("-lambdaLRAE"));

        if (argMap.containsKey("-Beta"))
            Beta = Double.parseDouble(argMap.get("-Beta"));

        if (argMap.containsKey("-trainModel"))
            trainModel = Boolean.parseBoolean(argMap.get("-trainModel"));

        if (argMap.containsKey("-WordMapFile")) {
            WordMapFile = argMap.get("-WordMapFile");
            if (!exists(WordMapFile)) {
                System.err.println("Your WordMapFile points to an invalid file!");
                exitOnReturn = true;
                printUsage();
                return;
            }
        }

        if (argMap.containsKey("-LabelMapFile")) {
            LabelMapFile = argMap.get("-LabelMapFile");
            if (!exists(LabelMapFile)) {
                System.err.println("Your LabelMapFile points to an invalid file!");
                exitOnReturn = true;
                printUsage();
                return;
            }
        }

        if (argMap.containsKey("-ModelFile"))
            ModelFile = argMap.get("-ModelFile");
        else {
            System.err.println("Please specify a ModelFile parameter.");
            exitOnReturn = true;
            printUsage();
            return;
        }

        if (argMap.containsKey("-ClassifierFile"))
            ClassifierFile = argMap.get("-ClassifierFile");
        else {
            System.err.println("Please specify a ClassifierFile parameter.");
            exitOnReturn = true;
            printUsage();
            return;
        }

        if (argMap.containsKey("-FeaturesOutputFile"))
            featuresOutputFile = argMap.get("-FeaturesOutputFile");

        if (argMap.containsKey("-TreeDumpDir")) {
            treeDumpDir = argMap.get("-TreeDumpDir");

            File treeDumpFile = new File(treeDumpDir);
            if (!treeDumpFile.exists())
                treeDumpFile.mkdir();
            else if (!treeDumpFile.isDirectory()) {
                System.err.println("TreeDumpDir file exists but it is not a directory.");
                exitOnReturn = true;
                printUsage();
            }
        }
        if (argMap.containsKey("-MatrixDumpDir")) {
            matrixDumpDir = argMap.get("-MatrixDumpDir");

            File matrixDumpFile = new File(matrixDumpDir);
            if (!matrixDumpFile.exists())
                matrixDumpFile.mkdir();
            else if (!matrixDumpFile.isDirectory()) {
                System.err.println("MatrixDumpDir file exists but it is not a directory.");
                exitOnReturn = true;
                printUsage();
            }
        }
        if (argMap.containsKey("-ProbabilitiesOutputFile"))
            ProbabilitiesOutputFile = argMap.get("-ProbabilitiesOutputFile");

        if (!trainModel && (ProbabilitiesOutputFile == null && featuresOutputFile == null)) {
            System.err.println("Please specify your output if you are not training.");
            exitOnReturn = true;
            printUsage();
            return;
        }

        if (argMap.containsKey("-NumCores")) {
            int numCores = Integer.parseInt(argMap.get("-NumCores"));
            Parallel.setPoolSize(numCores);
            ThreadPool.setPoolSize(numCores);
        }

        if (argMap.containsKey("--help") || argMap.containsKey("-h")) {
            exitOnReturn = true;
            printUsage();
            return;
        }

        if (argMap.containsKey("-ProcessedDataDir")) {
            dir = argMap.get("-ProcessedDataDir");
            Dataset = new MatProcessData(dir);
        } else if (argMap.containsKey("-DataDir")) {
            dir = argMap.get("-DataDir");
            ParsedReviewData Data = new ParsedReviewData(dir, minCount, WordMapFile, LabelMapFile);
            if (Data.isTestLablesKnown()) {
                isTestLabelsKnown = true;
                System.out.println("Test Lables known!");
            }
            Dataset = Data;
        } else
            Dataset = new MatProcessData(dir);

        catSize = Dataset.getCatSize() - 1;
        dictionarySize = Dataset.Vocab.size();
        hiddenSize = embeddingSize;
        visibleSize = embeddingSize;

        System.out.println("NumCategories : " + Dataset.getCatSize());
    }

    public void printUsage() {
        try {
            FileInputStream fstream = new FileInputStream("USAGE");
            DataInputStream in = new DataInputStream(fstream);
            BufferedReader br = new BufferedReader(new InputStreamReader(in));
            String strLine;

            while ((strLine = br.readLine()) != null) {
                System.out.println(strLine);
            }
            in.close();
        } catch (Exception e) {
            // Catch exception if any
            System.err.println("Error: " + e.getMessage());
        }
    }

    private boolean exists(String fileName) {
        return new File(fileName).exists();
    }

}
