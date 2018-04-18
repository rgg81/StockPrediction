package com.isaac.stock.predict;

import com.isaac.stock.model.RecurrentNets;
import com.isaac.stock.representation.BinanceDataSetIterator;
import com.isaac.stock.representation.PriceCategory;
import com.isaac.stock.representation.StockDataSetIterator;
import com.isaac.stock.utils.PlotUtil;
import javafx.util.Pair;
import org.datavec.api.transform.transform.doubletransform.MinMaxNormalizer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 * @author ZHANG HAO
 */
public class BinancePricePrediction {

    private static final Logger log = LoggerFactory.getLogger(BinancePricePrediction.class);

    public static void main (String[] args) throws IOException {

        int batchSize = 64; // mini-batch size
        int epochs = 4; // training epochs
//        int epochs = 1; // training epochs

        NormalizerStandardize normalizer = new NormalizerStandardize();

        log.info("Create dataSet iterator...");
        BinanceDataSetIterator trainData = new BinanceDataSetIterator(179,27);
        log.info("Load test dataset...");
        BinanceDataSetIterator testData = new BinanceDataSetIterator(188, 180);

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(trainData.inputColumns(), trainData.totalOutcomes());


        log.info("Normalizing..");
        normalizer.fit(trainData);
        trainData.reset();

        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);


        for (int i = 0; i < epochs; i++) {
            log.info("Training... epoch: {}", i);
            net.fit(trainData);

            Evaluation evaluation = net.evaluate(testData);
            System.out.println(evaluation.confusionToString());
            System.out.println(evaluation.stats());

            trainData.reset();
            testData.reset();
        }

        log.info("Saving model...");
        File locationToSave = new File("src/main/resources/StockPriceLSTM_.zip");
//        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        log.info("Load model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
//        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("Testing...");

        predictData(net, testData);

        log.info("Done...");
    }

    /** Predict one feature of a stock one-day ahead */
    private static void predictData (MultiLayerNetwork net, DataSetIterator testData) {


        INDArray guesses = net.output(testData, false);

        testData.reset();
        DataSet ds = testData.next();
        INDArray labels = ds.getLabels();

        INDArray realOutcomeIndex = Nd4j.argMax(labels, 1);
        INDArray guessIndex = Nd4j.argMax(guesses, 1);

        int nExamples = realOutcomeIndex.length();

        log.info("total Examples:{} printing first 100..", nExamples);
        for (int i = 0; i < 100; i++) {
            int actual = (int) realOutcomeIndex.getDouble(i);
            int predicted = (int) guessIndex.getDouble(i);
            log.info("actual:{} predicted:{}",actual,predicted);
        }

        log.info("shape info:{} rows:{} columns:{}", labels.shapeInfoToString());
        log.info("shape info row 0:{} rows:{} columns:{}", labels.getRow(0).shapeInfoToString(), labels.getRow(0).rows(), labels.getRow(0).columns());



    }



}
