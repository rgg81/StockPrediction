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


    private int predictLength = 1; // default 1, say, one day ahead prediction

    private int maxIteration;
    private int startIteration;

    private int currentIteration;

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
            int end = currentIteration + offsetIteration;
            CSVSequenceRecordReader trainFeaturesUp = new CSVSequenceRecordReader(1, ",");
            trainFeaturesUp.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/features-clean-%d.csv", currentIteration, end));
            CSVSequenceRecordReader trainLabelsUp = new CSVSequenceRecordReader(1, ",");
            trainLabelsUp.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/export-BTC-USDT-indicators-label-binance-up-%d.csv", currentIteration, end));
            SequenceRecordReaderDataSetIterator trainDataUp = new SequenceRecordReaderDataSetIterator(trainFeaturesUp, trainLabelsUp, miniBatchSize, 1, true);
            currentIteration = end + 1;
            return trainDataUp.next();
        } catch (Exception e) {
            throw new IllegalArgumentException("Error in parsing files", e);
        }
    }


    @Override public int totalExamples() { return miniBatchSize * (maxIteration-startIteration); }

    @Override public int inputColumns() {
        try{
            CSVSequenceRecordReader trainFeaturesUp = new CSVSequenceRecordReader(1, ",");
            trainFeaturesUp.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/features-clean-%d.csv", 0, 3));
            CSVSequenceRecordReader trainLabelsUp = new CSVSequenceRecordReader(1, ",");
            trainLabelsUp.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/export-BTC-USDT-indicators-label-binance-up-%d.csv", 0, 3));
            SequenceRecordReaderDataSetIterator trainDataUp = new SequenceRecordReaderDataSetIterator(trainFeaturesUp, trainLabelsUp, miniBatchSize, 1, true);
            return trainDataUp.inputColumns();
        } catch (Exception e){
            throw new IllegalArgumentException("Error in parsing files",e);
        }


    }

    @Override public int totalOutcomes() {

        return predictLength;
    }

    @Override public boolean resetSupported() { return false; }

    @Override public boolean asyncSupported() { return false; }

    @Override public void reset() { currentIteration = startIteration; }

    @Override public int batch() { return miniBatchSize; }

    @Override public int cursor() { return currentIteration; }

    @Override public int numExamples() { return totalExamples(); }

    @Override public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override public DataSetPreProcessor getPreProcessor() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override public List<String> getLabels() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override public boolean hasNext() { return currentIteration < maxIteration; }

    @Override public DataSet next() { return next(miniBatchSize); }
    



}
