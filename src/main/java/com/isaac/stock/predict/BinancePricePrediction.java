package com.isaac.stock.predict;

import com.isaac.stock.model.RecurrentNets;
import com.isaac.stock.representation.BinanceDataSetIterator;
import com.isaac.stock.representation.PriceCategory;
import com.isaac.stock.representation.StockDataSetIterator;
import com.isaac.stock.utils.PlotUtil;
import javafx.util.Pair;
import org.datavec.api.transform.transform.doubletransform.MinMaxNormalizer;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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

//        int batchSize = 64; // mini-batch size
        int epochs = 100; // training epochs

        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();

        log.info("Create dataSet iterator...");
        BinanceDataSetIterator trainData = new BinanceDataSetIterator(150,0);
        log.info("Load test dataset...");
        BinanceDataSetIterator testData = new BinanceDataSetIterator(170, 151);

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(trainData.inputColumns(), trainData.totalOutcomes());


        log.info("Normalizing..");
        normalizer.fit(trainData);
        trainData.reset();

        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);


        for (int i = 0; i < epochs; i++) {
            log.info("Training... epoch: {}", i);
            while (trainData.hasNext()) net.fit(trainData.next()); // fit model using mini-batch data
            RegressionEvaluation eval = net.evaluateRegression(testData);
            log.info(eval.stats());
            trainData.reset(); // reset iterator
            testData.reset();

            predictPriceOneAhead(net, testData, normalizer);
            testData.reset();

            net.rnnClearPreviousState(); // clear previous state
        }

        log.info("Saving model...");
        File locationToSave = new File("src/main/resources/StockPriceLSTM_.zip");
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        log.info("Load model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("Testing...");

        predictPriceOneAhead(net, testData, normalizer);

        log.info("Done...");
    }

    /** Predict one feature of a stock one-day ahead */
    private static void predictPriceOneAhead (MultiLayerNetwork net, DataSetIterator testData, NormalizerMinMaxScaler normalizer) {

        while (testData.hasNext()) {
            DataSet ds = testData.next();
            double[] output = net.output(ds.getFeatures(), false).data().asDouble();
            double[] labels = ds.getLabels().data().asDouble();

            //first 50
            for (int i = 0; i < 50; i++) {
                log.info("Data output:{} data labels:{}",output[i],labels[i]);
            }
            log.info("Data length:{}", output.length);

//            normalizer.revertLabels(output);
//            normalizer.revertLabels(labels);

        }

    }



}
