package com.isaac.stock.model;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by zhanghao on 26/7/17.
 * @author ZHANG HAO
 */
public class RecurrentNets {
	
//	private static final double learningRate = 0.05;
    private static final double learningRate = 0.1;
	private static final int iterations = 1;
	private static final int seed = 12345;

    private static final int lstmLayer1Size = 512;
    private static final int lstmLayer2Size = 256;
    private static final int denseLayerSize = 128;
    private static final double dropoutRatio = 0.2;
    private static final int truncatedBPTTLength = 140;

    public static MultiLayerNetwork buildLstmNetworks(int nIn, int nOut) {

        //Initialize the user interface backend
//        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true)
                .l2(1e-4)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(nIn)
                        .nOut(lstmLayer1Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(lstmLayer1Size)
                        .nOut(lstmLayer2Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                .layer(2, new DenseLayer.Builder()
                		.nIn(lstmLayer2Size)
                		.nOut(denseLayerSize)
                		.activation(Activation.RELU)
                		.build())
                .layer(3, new RnnOutputLayer.Builder()
                        .nIn(denseLayerSize)
                        .nOut(nOut)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(truncatedBPTTLength)
                .tBPTTBackwardLength(truncatedBPTTLength)
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        return net;
    }
}
