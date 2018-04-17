package com.isaac.stock.representation;

import com.google.common.collect.ImmutableMap;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 * @author ZHANG HAO
 */
public class BinanceDataSetIterator implements DataSetIterator {


    private static final Logger log = LoggerFactory.getLogger(BinanceDataSetIterator.class);
    private int predictLength = 1; // default 1, say, one day ahead prediction

    private int maxIteration;
    private int startIteration;

    private int currentIteration;
    private DataSetPreProcessor dataSetPreProcessor = null;

    private File baseDir = new File("/home/roberto/binance-data-train");
    private File featuresDir = new File(baseDir, "features");
    private File labelsDir = new File(baseDir, "labels");

    private int miniBatchSize = 472;
    private int offsetIteration = 90;

    public BinanceDataSetIterator(int maxIteration, int startIteration) {
        this.maxIteration = maxIteration;
        this.startIteration = startIteration;
        this.currentIteration = startIteration;
    }


    @Override
    public DataSet next(int num) {
        try {
            int end = Math.min(maxIteration,currentIteration + offsetIteration);
            log.info("currentIteration:{} end:{}",currentIteration,end);
            CSVSequenceRecordReader trainFeaturesUp = new CSVSequenceRecordReader(1, ",");
            trainFeaturesUp.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/features-clean-%d.csv", currentIteration, end));
            CSVSequenceRecordReader trainLabelsUp = new CSVSequenceRecordReader(1, ",");
            trainLabelsUp.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/export-BTC-USDT-indicators-label-binance-up-%d.csv", currentIteration, end));
            SequenceRecordReaderDataSetIterator trainDataUp = new SequenceRecordReaderDataSetIterator(trainFeaturesUp, trainLabelsUp, miniBatchSize, 9, false,
                    SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
            if(dataSetPreProcessor != null)
                trainDataUp.setPreProcessor(dataSetPreProcessor);
            currentIteration = end + 1;
            DataSet resultDs = trainDataUp.next();
            if(trainDataUp.hasNext()) {
                log.error("there is a next here... why????");
            }
            return resultDs;
        } catch (Exception e) {
            throw new IllegalArgumentException("Error in parsing files", e);
        }
    }


    @Override public int totalExamples() { return miniBatchSize * (maxIteration-startIteration); }

    @Override public int inputColumns() {
        try{
            CSVSequenceRecordReader trainFeaturesUp = new CSVSequenceRecordReader(1, ",");
            trainFeaturesUp.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/features-clean-%d.csv", 0, 10));
            CSVSequenceRecordReader trainLabelsUp = new CSVSequenceRecordReader(1, ",");
            trainLabelsUp.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/export-BTC-USDT-indicators-label-binance-up-%d.csv", 0, 10));
            SequenceRecordReaderDataSetIterator trainDataUp = new SequenceRecordReaderDataSetIterator(trainFeaturesUp, trainLabelsUp, miniBatchSize, 9, false);
            return trainDataUp.inputColumns();
        } catch (Exception e){
            throw new IllegalArgumentException("Error in parsing files",e);
        }


    }

    @Override public int totalOutcomes() {

        return 9;
    }

    @Override public boolean resetSupported() { return false; }

    @Override public boolean asyncSupported() { return false; }

    @Override public void reset() { currentIteration = startIteration; }

    @Override public int batch() { return miniBatchSize; }

    @Override public int cursor() { return currentIteration; }

    @Override public int numExamples() { return totalExamples(); }

    @Override public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        this.dataSetPreProcessor = dataSetPreProcessor;
    }

    @Override public DataSetPreProcessor getPreProcessor() { return dataSetPreProcessor; }

    @Override public List<String> getLabels() { return Arrays.asList("1","2","3","4","0","5","6","7","8"); }

    @Override public boolean hasNext() { return currentIteration < maxIteration; }

    @Override public DataSet next() { return next(miniBatchSize); }
    



}
