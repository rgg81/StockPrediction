package com.isaac.stock.predict;

import com.isaac.stock.model.RecurrentNets;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class BinancePricePrediction2 {

    private static final Logger LOGGER = LoggerFactory.getLogger(BinancePricePrediction2.class);

    public static void main(String[] args) throws Exception {
        File baseDir = new File("/home/roberto/binance-data-train");
        File featuresDir = new File(baseDir, "features");
        File labelsDir = new File(baseDir, "labels");


        int miniBatchSize = 472;

        CSVSequenceRecordReader trainFeaturesUp = new CSVSequenceRecordReader(1, ",");
        trainFeaturesUp.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/features-clean-%d.csv", 0, 150));
        CSVSequenceRecordReader trainLabelsUp = new CSVSequenceRecordReader(1, ",");
        trainLabelsUp.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/export-BTC-USDT-indicators-label-binance-up-%d.csv", 0, 150));
        SequenceRecordReaderDataSetIterator trainDataUp = new SequenceRecordReaderDataSetIterator(trainFeaturesUp, trainLabelsUp, miniBatchSize, -1, true,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainDataUp);              //Collect training data statistics
        trainDataUp.reset();

        CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/features-clean-%d.csv", 151, 159));
        CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader(1, ",");
        testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/export-BTC-USDT-indicators-label-binance-up-%d.csv", 151, 159));
        SequenceRecordReaderDataSetIterator testDataIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        trainDataUp.setPreProcessor(normalizer);
        testDataIter.setPreProcessor(normalizer);



        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(trainDataUp.inputColumns(),1);

        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 5;

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainDataUp);
            trainDataUp.reset();
            LOGGER.info("Epoch " + i + " complete. Time series evaluation:");

            //Run regression evaluation on our single column input
            RegressionEvaluation evaluation = new RegressionEvaluation(1);

            //Run evaluation. This is on 25k reviews, so can take some time
            while (testDataIter.hasNext()) {
                DataSet t = testDataIter.next();
                INDArray features = t.getFeatures();
                INDArray lables = t.getLabels();
                INDArray predicted = net.output(features, false);

                evaluation.evalTimeSeries(lables, predicted);
            }

            System.out.println(evaluation.stats());

            testDataIter.reset();
        }

        //Init rrnTimeStemp with train data and predict test data

//        while (trainDataUp.hasNext()) {
//            DataSet t = trainDataUp.next();
//            net.rnnTimeStep(t.getFeatures());
//        }

        trainDataUp.reset();

        DataSet t = testDataIter.next();
        INDArray predicted = net.rnnTimeStep(t.getFeatures());
        normalizer.revertLabels(predicted);
        normalizer.revertLabels(t.getLabels());

        int nRows = predicted.shape()[2];

        for (int i = 0; i < nRows; i++) {
            LOGGER.info("Predicted:{} expected:{}", predicted.getDouble(i),t.getLabels().getDouble(i));
        }

    }




    //Normalize the training data

}
