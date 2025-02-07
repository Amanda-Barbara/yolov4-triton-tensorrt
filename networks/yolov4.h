#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cmath>

#include "../utils/weights.h"

using namespace nvinfer1;

//#define USE_FP16

namespace yolov4 {

    // stuff we know about the network and the input/output blobs
    static const int INPUT_H = 608;
    static const int INPUT_W = 608;
    static const int CLASS_NUM = 5;

    static const int YOLO_FACTOR_1 = 8;
    static const std::vector<float> YOLO_ANCHORS_1 = { 12,16, 19,36, 40,28 };
    static const float YOLO_SCALE_XY_1 = 1.2f;
    static const int YOLO_NEWCOORDS_1 = 0;

    static const int YOLO_FACTOR_2 = 16;
    static const std::vector<float> YOLO_ANCHORS_2 = { 36,75, 76,55, 72,146 };
    static const float YOLO_SCALE_XY_2 = 1.1f;
    static const int YOLO_NEWCOORDS_2 = 0;

    static const int YOLO_FACTOR_3 = 32;
    static const std::vector<float> YOLO_ANCHORS_3 = { 142,110, 192,243, 459,401 };
    static const float YOLO_SCALE_XY_3 = 1.05f;
    static const int YOLO_NEWCOORDS_3 = 0;

    const char* INPUT_BLOB_NAME = "input";
    const char* OUTPUT_BLOB_NAME = "detections";

    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
        float *gamma = (float*)weightMap[lname + ".weight"].values;
        float *beta = (float*)weightMap[lname + ".bias"].values;
        float *mean = (float*)weightMap[lname + ".running_mean"].values;
        float *var = (float*)weightMap[lname + ".running_var"].values;
        int len = weightMap[lname + ".running_var"].count;

        float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            scval[i] = gamma[i] / sqrt(var[i] + eps);
        }
        Weights scale{DataType::kFLOAT, scval, len};
        
        float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        }
        Weights shift{DataType::kFLOAT, shval, len};

        float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            pval[i] = 1.0;
        }
        Weights power{DataType::kFLOAT, pval, len};

        weightMap[lname + ".scale"] = scale;
        weightMap[lname + ".shift"] = shift;
        weightMap[lname + ".power"] = power;
        IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
        assert(scale_1);
        return scale_1;
    }

    ILayer* convBnMish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["model." + std::to_string(linx) + ".conv.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{s, s});
        conv1->setPaddingNd(DimsHW{p, p});

        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "model." + std::to_string(linx) + ".bn", 1e-4);

        auto mish_softplus = network->addActivation(*bn1->getOutput(0), ActivationType::kSOFTPLUS);
        auto mish_tanh = network->addActivation(*mish_softplus->getOutput(0), ActivationType::kTANH);
        auto mish_mul = network->addElementWise(*mish_tanh->getOutput(0), *bn1->getOutput(0), ElementWiseOperation::kPROD);

        return mish_mul;
    }

    ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["model." + std::to_string(linx) + ".conv.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{s, s});
        conv1->setPaddingNd(DimsHW{p, p});

        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "model." + std::to_string(linx) + ".bn", 1e-4);

        auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
        lr->setAlpha(0.1);

        return lr;
    }

    IPluginV2Layer * yoloLayer(INetworkDefinition *network, ITensor& input, int inputWidth, int inputHeight, int widthFactor, int heightFactor, int numClasses, const std::vector<float>& anchors, float scaleXY, int newCoords) {
        auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
        
        int yoloWidth = inputWidth / widthFactor;
        int yoloHeight = inputHeight / heightFactor;
        int numAnchors = anchors.size() / 2;

        PluginFieldCollection pluginData;
        std::vector<PluginField> pluginFields;
        pluginFields.emplace_back(PluginField("yoloWidth", &yoloWidth, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("yoloHeight", &yoloHeight, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("numAnchors", &numAnchors, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("numClasses", &numClasses, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("inputMultiplier", &widthFactor, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("anchors", anchors.data(), PluginFieldType::kFLOAT32, anchors.size()));
        pluginFields.emplace_back(PluginField("scaleXY", &scaleXY, PluginFieldType::kFLOAT32, 1));
        pluginFields.emplace_back(PluginField("newCoords", &newCoords, PluginFieldType::kINT32, 1));
        pluginData.nbFields = pluginFields.size();
        pluginData.fields = pluginFields.data();

        IPluginV2 *plugin = creator->createPlugin("YoloLayer_TRT", &pluginData);
        ITensor* inputTensors[] = { &input };
        return network->addPluginV2(inputTensors, 1, *plugin);
    }

    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string &weightsPath) {
        INetworkDefinition* network = builder->createNetworkV2(0U);

        // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
        ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
        assert(data);

        std::map<std::string, Weights> weightMap = loadWeights(weightsPath);
        Weights emptywts{DataType::kFLOAT, nullptr, 0};

        // define each layer.
        auto l0 = convBnMish(network, weightMap, *data, 32, 3, 1, 1, 0);
        auto l1 = convBnMish(network, weightMap, *l0->getOutput(0), 64, 3, 2, 1, 1);
        auto l2 = convBnMish(network, weightMap, *l1->getOutput(0), 64, 1, 1, 0, 2);
        auto l3 = l1;
        auto l4 = convBnMish(network, weightMap, *l3->getOutput(0), 64, 1, 1, 0, 4);
        auto l5 = convBnMish(network, weightMap, *l4->getOutput(0), 32, 1, 1, 0, 5);
        auto l6 = convBnMish(network, weightMap, *l5->getOutput(0), 64, 3, 1, 1, 6);
        auto ew7 = network->addElementWise(*l6->getOutput(0), *l4->getOutput(0), ElementWiseOperation::kSUM);
        auto l8 = convBnMish(network, weightMap, *ew7->getOutput(0), 64, 1, 1, 0, 8);

        ITensor* inputTensors9[] = {l8->getOutput(0), l2->getOutput(0)};
        auto cat9 = network->addConcatenation(inputTensors9, 2);

        auto l10 = convBnMish(network, weightMap, *cat9->getOutput(0), 64, 1, 1, 0, 10);
        auto l11 = convBnMish(network, weightMap, *l10->getOutput(0), 128, 3, 2, 1, 11);
        auto l12 = convBnMish(network, weightMap, *l11->getOutput(0), 64, 1, 1, 0, 12);
        auto l13 = l11;
        auto l14 = convBnMish(network, weightMap, *l13->getOutput(0), 64, 1, 1, 0, 14);
        auto l15 = convBnMish(network, weightMap, *l14->getOutput(0), 64, 1, 1, 0, 15);
        auto l16 = convBnMish(network, weightMap, *l15->getOutput(0), 64, 3, 1, 1, 16);
        auto ew17 = network->addElementWise(*l16->getOutput(0), *l14->getOutput(0), ElementWiseOperation::kSUM);
        auto l18 = convBnMish(network, weightMap, *ew17->getOutput(0), 64, 1, 1, 0, 18);
        auto l19 = convBnMish(network, weightMap, *l18->getOutput(0), 64, 3, 1, 1, 19);
        auto ew20 = network->addElementWise(*l19->getOutput(0), *ew17->getOutput(0), ElementWiseOperation::kSUM);
        auto l21 = convBnMish(network, weightMap, *ew20->getOutput(0), 64, 1, 1, 0, 21);

        ITensor* inputTensors22[] = {l21->getOutput(0), l12->getOutput(0)};
        auto cat22 = network->addConcatenation(inputTensors22, 2);

        auto l23 = convBnMish(network, weightMap, *cat22->getOutput(0), 128, 1, 1, 0, 23);
        auto l24 = convBnMish(network, weightMap, *l23->getOutput(0), 256, 3, 2, 1, 24);
        auto l25 = convBnMish(network, weightMap, *l24->getOutput(0), 128, 1, 1, 0, 25);
        auto l26 = l24;
        auto l27 = convBnMish(network, weightMap, *l26->getOutput(0), 128, 1, 1, 0, 27);
        auto l28 = convBnMish(network, weightMap, *l27->getOutput(0), 128, 1, 1, 0, 28);
        auto l29 = convBnMish(network, weightMap, *l28->getOutput(0), 128, 3, 1, 1, 29);
        auto ew30 = network->addElementWise(*l29->getOutput(0), *l27->getOutput(0), ElementWiseOperation::kSUM);
        auto l31 = convBnMish(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
        auto l32 = convBnMish(network, weightMap, *l31->getOutput(0), 128, 3, 1, 1, 32);
        auto ew33 = network->addElementWise(*l32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
        auto l34 = convBnMish(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
        auto l35 = convBnMish(network, weightMap, *l34->getOutput(0), 128, 3, 1, 1, 35);
        auto ew36 = network->addElementWise(*l35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
        auto l37 = convBnMish(network, weightMap, *ew36->getOutput(0), 128, 1, 1, 0, 37);
        auto l38 = convBnMish(network, weightMap, *l37->getOutput(0), 128, 3, 1, 1, 38);
        auto ew39 = network->addElementWise(*l38->getOutput(0), *ew36->getOutput(0), ElementWiseOperation::kSUM);
        auto l40 = convBnMish(network, weightMap, *ew39->getOutput(0), 128, 1, 1, 0, 40);
        auto l41 = convBnMish(network, weightMap, *l40->getOutput(0), 128, 3, 1, 1, 41);
        auto ew42 = network->addElementWise(*l41->getOutput(0), *ew39->getOutput(0), ElementWiseOperation::kSUM);
        auto l43 = convBnMish(network, weightMap, *ew42->getOutput(0), 128, 1, 1, 0, 43);
        auto l44 = convBnMish(network, weightMap, *l43->getOutput(0), 128, 3, 1, 1, 44);
        auto ew45 = network->addElementWise(*l44->getOutput(0), *ew42->getOutput(0), ElementWiseOperation::kSUM);
        auto l46 = convBnMish(network, weightMap, *ew45->getOutput(0), 128, 1, 1, 0, 46);
        auto l47 = convBnMish(network, weightMap, *l46->getOutput(0), 128, 3, 1, 1, 47);
        auto ew48 = network->addElementWise(*l47->getOutput(0), *ew45->getOutput(0), ElementWiseOperation::kSUM);
        auto l49 = convBnMish(network, weightMap, *ew48->getOutput(0), 128, 1, 1, 0, 49);
        auto l50 = convBnMish(network, weightMap, *l49->getOutput(0), 128, 3, 1, 1, 50);
        auto ew51 = network->addElementWise(*l50->getOutput(0), *ew48->getOutput(0), ElementWiseOperation::kSUM);
        auto l52 = convBnMish(network, weightMap, *ew51->getOutput(0), 128, 1, 1, 0, 52);

        ITensor* inputTensors53[] = {l52->getOutput(0), l25->getOutput(0)};
        auto cat53 = network->addConcatenation(inputTensors53, 2);

        auto l54 = convBnMish(network, weightMap, *cat53->getOutput(0), 256, 1, 1, 0, 54);
        auto l55 = convBnMish(network, weightMap, *l54->getOutput(0), 512, 3, 2, 1, 55);
        auto l56 = convBnMish(network, weightMap, *l55->getOutput(0), 256, 1, 1, 0, 56);
        auto l57 = l55;
        auto l58 = convBnMish(network, weightMap, *l57->getOutput(0), 256, 1, 1, 0, 58);
        auto l59 = convBnMish(network, weightMap, *l58->getOutput(0), 256, 1, 1, 0, 59);
        auto l60 = convBnMish(network, weightMap, *l59->getOutput(0), 256, 3, 1, 1, 60);
        auto ew61 = network->addElementWise(*l60->getOutput(0), *l58->getOutput(0), ElementWiseOperation::kSUM);
        auto l62 = convBnMish(network, weightMap, *ew61->getOutput(0), 256, 1, 1, 0, 62);
        auto l63 = convBnMish(network, weightMap, *l62->getOutput(0), 256, 3, 1, 1, 63);
        auto ew64 = network->addElementWise(*l63->getOutput(0), *ew61->getOutput(0), ElementWiseOperation::kSUM);
        auto l65 = convBnMish(network, weightMap, *ew64->getOutput(0), 256, 1, 1, 0, 65);
        auto l66 = convBnMish(network, weightMap, *l65->getOutput(0), 256, 3, 1, 1, 66);
        auto ew67 = network->addElementWise(*l66->getOutput(0), *ew64->getOutput(0), ElementWiseOperation::kSUM);
        auto l68 = convBnMish(network, weightMap, *ew67->getOutput(0), 256, 1, 1, 0, 68);
        auto l69 = convBnMish(network, weightMap, *l68->getOutput(0), 256, 3, 1, 1, 69);
        auto ew70 = network->addElementWise(*l69->getOutput(0), *ew67->getOutput(0), ElementWiseOperation::kSUM);
        auto l71 = convBnMish(network, weightMap, *ew70->getOutput(0), 256, 1, 1, 0, 71);
        auto l72 = convBnMish(network, weightMap, *l71->getOutput(0), 256, 3, 1, 1, 72);
        auto ew73 = network->addElementWise(*l72->getOutput(0), *ew70->getOutput(0), ElementWiseOperation::kSUM);
        auto l74 = convBnMish(network, weightMap, *ew73->getOutput(0), 256, 1, 1, 0, 74);
        auto l75 = convBnMish(network, weightMap, *l74->getOutput(0), 256, 3, 1, 1, 75);
        auto ew76 = network->addElementWise(*l75->getOutput(0), *ew73->getOutput(0), ElementWiseOperation::kSUM);
        auto l77 = convBnMish(network, weightMap, *ew76->getOutput(0), 256, 1, 1, 0, 77);
        auto l78 = convBnMish(network, weightMap, *l77->getOutput(0), 256, 3, 1, 1, 78);
        auto ew79 = network->addElementWise(*l78->getOutput(0), *ew76->getOutput(0), ElementWiseOperation::kSUM);
        auto l80 = convBnMish(network, weightMap, *ew79->getOutput(0), 256, 1, 1, 0, 80);
        auto l81 = convBnMish(network, weightMap, *l80->getOutput(0), 256, 3, 1, 1, 81);
        auto ew82 = network->addElementWise(*l81->getOutput(0), *ew79->getOutput(0), ElementWiseOperation::kSUM);
        auto l83 = convBnMish(network, weightMap, *ew82->getOutput(0), 256, 1, 1, 0, 83);

        ITensor* inputTensors84[] = {l83->getOutput(0), l56->getOutput(0)};
        auto cat84 = network->addConcatenation(inputTensors84, 2);

        auto l85 = convBnMish(network, weightMap, *cat84->getOutput(0), 512, 1, 1, 0, 85);
        auto l86 = convBnMish(network, weightMap, *l85->getOutput(0), 1024, 3, 2, 1, 86);
        auto l87 = convBnMish(network, weightMap, *l86->getOutput(0), 512, 1, 1, 0, 87);
        auto l88 = l86;
        auto l89 = convBnMish(network, weightMap, *l88->getOutput(0), 512, 1, 1, 0, 89);
        auto l90 = convBnMish(network, weightMap, *l89->getOutput(0), 512, 1, 1, 0, 90);
        auto l91 = convBnMish(network, weightMap, *l90->getOutput(0), 512, 3, 1, 1, 91);
        auto ew92 = network->addElementWise(*l91->getOutput(0), *l89->getOutput(0), ElementWiseOperation::kSUM);
        auto l93 = convBnMish(network, weightMap, *ew92->getOutput(0), 512, 1, 1, 0, 93);
        auto l94 = convBnMish(network, weightMap, *l93->getOutput(0), 512, 3, 1, 1, 94);
        auto ew95 = network->addElementWise(*l94->getOutput(0), *ew92->getOutput(0), ElementWiseOperation::kSUM);
        auto l96 = convBnMish(network, weightMap, *ew95->getOutput(0), 512, 1, 1, 0, 96);
        auto l97 = convBnMish(network, weightMap, *l96->getOutput(0), 512, 3, 1, 1, 97);
        auto ew98 = network->addElementWise(*l97->getOutput(0), *ew95->getOutput(0), ElementWiseOperation::kSUM);
        auto l99 = convBnMish(network, weightMap, *ew98->getOutput(0), 512, 1, 1, 0, 99);
        auto l100 = convBnMish(network, weightMap, *l99->getOutput(0), 512, 3, 1, 1, 100);
        auto ew101 = network->addElementWise(*l100->getOutput(0), *ew98->getOutput(0), ElementWiseOperation::kSUM);
        auto l102 = convBnMish(network, weightMap, *ew101->getOutput(0), 512, 1, 1, 0, 102);

        ITensor* inputTensors103[] = {l102->getOutput(0), l87->getOutput(0)};
        auto cat103 = network->addConcatenation(inputTensors103, 2);

        auto l104 = convBnMish(network, weightMap, *cat103->getOutput(0), 1024, 1, 1, 0, 104);

        // ---------
        auto l105 = convBnLeaky(network, weightMap, *l104->getOutput(0), 512, 1, 1, 0, 105);
        auto l106 = convBnLeaky(network, weightMap, *l105->getOutput(0), 1024, 3, 1, 1, 106);
        auto l107 = convBnLeaky(network, weightMap, *l106->getOutput(0), 512, 1, 1, 0, 107);

        auto pool108 = network->addPoolingNd(*l107->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
        pool108->setPaddingNd(DimsHW{2, 2});
        pool108->setStrideNd(DimsHW{1, 1});

        auto l109 = l107;

        auto pool110 = network->addPoolingNd(*l109->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
        pool110->setPaddingNd(DimsHW{4, 4});
        pool110->setStrideNd(DimsHW{1, 1});

        auto l111 = l107;

        auto pool112 = network->addPoolingNd(*l111->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
        pool112->setPaddingNd(DimsHW{6, 6});
        pool112->setStrideNd(DimsHW{1, 1});

        ITensor* inputTensors113[] = {pool112->getOutput(0), pool110->getOutput(0), pool108->getOutput(0), l107->getOutput(0)};
        auto cat113 = network->addConcatenation(inputTensors113, 4);

        auto l114 = convBnLeaky(network, weightMap, *cat113->getOutput(0), 512, 1, 1, 0, 114);
        auto l115 = convBnLeaky(network, weightMap, *l114->getOutput(0), 1024, 3, 1, 1, 115);
        auto l116 = convBnLeaky(network, weightMap, *l115->getOutput(0), 512, 1, 1, 0, 116);
        auto l117 = convBnLeaky(network, weightMap, *l116->getOutput(0), 256, 1, 1, 0, 117);

        float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
        for (int i = 0; i < 256 * 2 * 2; i++) {
            deval[i] = 1.0;
        }
        Weights deconvwts118{DataType::kFLOAT, deval, 256 * 2 * 2};
        IDeconvolutionLayer* deconv118 = network->addDeconvolutionNd(*l117->getOutput(0), 256, DimsHW{2, 2}, deconvwts118, emptywts);
        assert(deconv118);
        deconv118->setStrideNd(DimsHW{2, 2});
        deconv118->setNbGroups(256);
        weightMap["deconv118"] = deconvwts118;

        auto l119 = l85;
        auto l120 = convBnLeaky(network, weightMap, *l119->getOutput(0), 256, 1, 1, 0, 120);

        ITensor* inputTensors121[] = {l120->getOutput(0), deconv118->getOutput(0)};
        auto cat121 = network->addConcatenation(inputTensors121, 2);

        auto l122 = convBnLeaky(network, weightMap, *cat121->getOutput(0), 256, 1, 1, 0, 122);
        auto l123 = convBnLeaky(network, weightMap, *l122->getOutput(0), 512, 3, 1, 1, 123);
        auto l124 = convBnLeaky(network, weightMap, *l123->getOutput(0), 256, 1, 1, 0, 124);
        auto l125 = convBnLeaky(network, weightMap, *l124->getOutput(0), 512, 3, 1, 1, 125);
        auto l126 = convBnLeaky(network, weightMap, *l125->getOutput(0), 256, 1, 1, 0, 126);
        auto l127 = convBnLeaky(network, weightMap, *l126->getOutput(0), 128, 1, 1, 0, 127);

        Weights deconvwts128{DataType::kFLOAT, deval, 128 * 2 * 2};
        IDeconvolutionLayer* deconv128 = network->addDeconvolutionNd(*l127->getOutput(0), 128, DimsHW{2, 2}, deconvwts128, emptywts);
        assert(deconv128);
        deconv128->setStrideNd(DimsHW{2, 2});
        deconv128->setNbGroups(128);

        auto l129 = l54;
        auto l130 = convBnLeaky(network, weightMap, *l129->getOutput(0), 128, 1, 1, 0, 130);

        ITensor* inputTensors131[] = {l130->getOutput(0), deconv128->getOutput(0)};
        auto cat131 = network->addConcatenation(inputTensors131, 2);

        auto l132 = convBnLeaky(network, weightMap, *cat131->getOutput(0), 128, 1, 1, 0, 132);
        auto l133 = convBnLeaky(network, weightMap, *l132->getOutput(0), 256, 3, 1, 1, 133);
        auto l134 = convBnLeaky(network, weightMap, *l133->getOutput(0), 128, 1, 1, 0, 134);
        auto l135 = convBnLeaky(network, weightMap, *l134->getOutput(0), 256, 3, 1, 1, 135);
        auto l136 = convBnLeaky(network, weightMap, *l135->getOutput(0), 128, 1, 1, 0, 136);
        auto l137 = convBnLeaky(network, weightMap, *l136->getOutput(0), 256, 3, 1, 1, 137);
        IConvolutionLayer* conv138 = network->addConvolutionNd(*l137->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.138.conv.weight"], weightMap["model.138.conv.bias"]);
        assert(conv138);

        // 139 is yolo layer
        auto yolo139 = yoloLayer(network, *conv138->getOutput(0), INPUT_W, INPUT_H, YOLO_FACTOR_1, YOLO_FACTOR_1, CLASS_NUM, YOLO_ANCHORS_1, YOLO_SCALE_XY_1, YOLO_NEWCOORDS_1);

        auto l140 = l136;
        auto l141 = convBnLeaky(network, weightMap, *l140->getOutput(0), 256, 3, 2, 1, 141);

        ITensor* inputTensors142[] = {l141->getOutput(0), l126->getOutput(0)};
        auto cat142 = network->addConcatenation(inputTensors142, 2);

        auto l143 = convBnLeaky(network, weightMap, *cat142->getOutput(0), 256, 1, 1, 0, 143);
        auto l144 = convBnLeaky(network, weightMap, *l143->getOutput(0), 512, 3, 1, 1, 144);
        auto l145 = convBnLeaky(network, weightMap, *l144->getOutput(0), 256, 1, 1, 0, 145);
        auto l146 = convBnLeaky(network, weightMap, *l145->getOutput(0), 512, 3, 1, 1, 146);
        auto l147 = convBnLeaky(network, weightMap, *l146->getOutput(0), 256, 1, 1, 0, 147);
        auto l148 = convBnLeaky(network, weightMap, *l147->getOutput(0), 512, 3, 1, 1, 148);
        IConvolutionLayer* conv149 = network->addConvolutionNd(*l148->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.149.conv.weight"], weightMap["model.149.conv.bias"]);
        assert(conv149);

        // 150 is yolo layer
        auto yolo150 = yoloLayer(network, *conv149->getOutput(0), INPUT_W, INPUT_H, YOLO_FACTOR_2, YOLO_FACTOR_2, CLASS_NUM, YOLO_ANCHORS_2, YOLO_SCALE_XY_2, YOLO_NEWCOORDS_2);

        auto l151 = l147;
        auto l152 = convBnLeaky(network, weightMap, *l151->getOutput(0), 512, 3, 2, 1, 152);

        ITensor* inputTensors153[] = {l152->getOutput(0), l116->getOutput(0)};
        auto cat153 = network->addConcatenation(inputTensors153, 2);

        auto l154 = convBnLeaky(network, weightMap, *cat153->getOutput(0), 512, 1, 1, 0, 154);
        auto l155 = convBnLeaky(network, weightMap, *l154->getOutput(0), 1024, 3, 1, 1, 155);
        auto l156 = convBnLeaky(network, weightMap, *l155->getOutput(0), 512, 1, 1, 0, 156);
        auto l157 = convBnLeaky(network, weightMap, *l156->getOutput(0), 1024, 3, 1, 1, 157);
        auto l158 = convBnLeaky(network, weightMap, *l157->getOutput(0), 512, 1, 1, 0, 158);
        auto l159 = convBnLeaky(network, weightMap, *l158->getOutput(0), 1024, 3, 1, 1, 159);
        IConvolutionLayer* conv160 = network->addConvolutionNd(*l159->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.160.conv.weight"], weightMap["model.160.conv.bias"]);
        assert(conv160);

        // 161 is yolo layer
        auto yolo161 = yoloLayer(network, *conv160->getOutput(0), INPUT_W, INPUT_H, YOLO_FACTOR_3, YOLO_FACTOR_3, CLASS_NUM, YOLO_ANCHORS_3, YOLO_SCALE_XY_3, YOLO_NEWCOORDS_3);
        
        ITensor* inputTensors162[] = {yolo139->getOutput(0), yolo150->getOutput(0), yolo161->getOutput(0)};
        auto cat162 = network->addConcatenation(inputTensors162, 3);
        cat162->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        network->markOutput(*cat162->getOutput(0));

        // Build engine
        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    #ifdef USE_FP16
        config->setFlag(BuilderFlag::kFP16);
    #else
        config->setFlag(BuilderFlag::kTF32);
    #endif
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Don't need the network any more
        network->destroy();

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*) (mem.second.values));
        }

        return engine;
    }
}