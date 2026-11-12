using CNTK;
using System.Diagnostics;

namespace beadando
{
    class BatchTraining
    {
        DeviceDescriptor device = DeviceDescriptor.CPUDevice;
        DataSet trainDataset { get; set; }
        DataSet validationDataset { get; set; }

        const int hiddenNeuronCount = 64;

        const int epochCount = 5;

        const int batchSize = 128;

        readonly Variable x;
        readonly Function y;

        public BatchTraining(DataSet train_dataset, DataSet validation_dataset)
        {
            trainDataset = train_dataset;
            validationDataset = validation_dataset;

            x = Variable.InputVariable(new int[] { trainDataset.InputSize }, DataType.Float);
            Parameter w1 = new Parameter(new int[] { hiddenNeuronCount, trainDataset.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter b = new Parameter(new int[] { hiddenNeuronCount }, DataType.Float, 0);
            Parameter w2 = new Parameter(new int[] { trainDataset.OutputSize, hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter b2 = new Parameter(new int[] { trainDataset.OutputSize }, DataType.Float, 0);
            y = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w2, CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w1, x), b))), b2));
        }

        public void TrainAndValidate()
        {

            //ki kell egészíteni a gráfot
            Variable yt = Variable.InputVariable(new int[] {trainDataset.OutputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            //mivel tanítunk
            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            for (int epochI = 1; epochI <= epochCount; epochI++)
            {

                double sumLoss = 0;
                double sumEval = 0;
                int samplesSeen = 0;

                trainDataset.Shuffle();
                for (int start = 0; start < trainDataset.Count; start += batchSize)
                {
                    int currentBatchSize = Math.Min(batchSize, trainDataset.Count - start);

                    int inputStart = start * trainDataset.InputSize;
                    int inputCount = currentBatchSize * trainDataset.InputSize;

                    int outputStart = start * trainDataset.OutputSize;
                    int outputCount = currentBatchSize * trainDataset.OutputSize;

                    Value x_val = Value.CreateBatch(x.Shape, trainDataset.Input.GetRange(inputStart, inputCount), device);
                    Value yt_val = Value.CreateBatch(yt.Shape, trainDataset.Output.GetRange(outputStart, outputCount), device);

                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        {x, x_val },
                        {yt, yt_val }
                    };

                    bool isSweepEnd = start + currentBatchSize >= trainDataset.Count;

                    trainer.TrainMinibatch(inputDataMap, isSweepEnd, device);

                    int samples = (int)trainer.PreviousMinibatchSampleCount();

                    sumLoss += trainer.PreviousMinibatchLossAverage() * samples;
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * samples;

                    samplesSeen += samples;
                }


                ValidationResult validationResult = new ValidationResult();

                for (int start = 0; start < validationDataset.Count; start += batchSize)
                {

                    int currentBatchSize = Math.Min(batchSize, validationDataset.Count - start);

                    int inputStart = start * validationDataset.InputSize;

                    int inputCount = currentBatchSize * validationDataset.InputSize;

                    int outputStart = start * validationDataset.OutputSize;

                    int outputCount = currentBatchSize * validationDataset.OutputSize;

                    Value x_val = Value.CreateBatch(x.Shape, validationDataset.Input.GetRange(inputStart, inputCount), device);

                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_val },
                    };

                    var outputDataMap = new Dictionary<Variable, Value>()
                    {
                        { y, null },
                    };

                    y.Evaluate(inputDataMap, outputDataMap, device);

                    var predictions = outputDataMap[y].GetDenseData<float>(y);

                    for (int i = 0; i < currentBatchSize; i++)
                    {

                        double probability = predictions[i][0];

                        // clamp, hogy ne legyen log(0)
                        probability = Math.Clamp(probability, 1e-7, 1.0 - 1e-7);

                        int predicted = probability >= 0.5 ? 1 : 0;
                        int actual = (int)validationDataset.Output[outputStart + i];

                        validationResult.Loss += -(actual * Math.Log(probability) + (1 - actual) * Math.Log(1 - probability));

                        if (predicted == 1 && actual == 1)
                            validationResult.TP++;
                        else if (predicted == 0 && actual == 0)
                            validationResult.TN++;
                        else if (predicted == 1 && actual == 0)
                            validationResult.FP++;
                        else
                            validationResult.FN++;
                    }

                }
                int total = validationResult.TP + validationResult.TN + validationResult.FP + validationResult.FN;
                validationResult.Loss /= total;
                validationResult.Accuracy = (double)(validationResult.TP + validationResult.TN) / total;
                validationResult.Precision = (double)validationResult.TP / (validationResult.TP + validationResult.FP);
                validationResult.Recall = (double)validationResult.TP / (validationResult.TP + validationResult.FN);
                validationResult.F1 = 2 * validationResult.Precision * validationResult.Recall / (validationResult.Precision + validationResult.Recall);

                Console.WriteLine(String.Format("{0}\n", epochI));
                Console.WriteLine("TRAINING\n");
                Console.WriteLine(String.Format("train loss:{0}\ttrain eval:{1}", sumLoss / samplesSeen, sumEval / samplesSeen));
                Console.WriteLine();
                Console.WriteLine("VALIDATION\n");
                Console.WriteLine(String.Format("validation loss:{0}", validationResult.Loss));
                Console.WriteLine(String.Format("Accuracy: {0}\t Precision: {1}\t Recall: {2}\t F1: {3}", validationResult.Accuracy, validationResult.Precision, validationResult.Recall, validationResult.F1));
                Console.WriteLine(String.Format("TP: {0}\t TN: {1}\t FP: {2}\t FN: {3}\t", validationResult.TP, validationResult.TN, validationResult.FP, validationResult.FN));

            }
        }



        public struct ValidationResult
        {
            public double Precision;
            public double Recall;
            public double F1;
            public double Accuracy;
            public double Loss;
            public int TP;
            public int TN;
            public int FP;
            public int FN;


        }

    }
}
