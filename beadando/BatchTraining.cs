using CNTK;

namespace beadando
{
    class BatchTraining
    {
        DeviceDescriptor device = DeviceDescriptor.CPUDevice;
        public DataSet dataset { get; private set; }
        const int hiddenNeuronCount = 64;

        const int epochCount = 5;

        const int batchSize = 128;

        readonly Variable x;
        readonly Function y;

        public BatchTraining(DataSet dataset)
        {
            this.dataset = dataset;

            x = Variable.InputVariable(new int[] { this.dataset.InputSize }, DataType.Float);
            Parameter w1 = new Parameter(new int[] { hiddenNeuronCount, this.dataset.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter b = new Parameter(new int[] { hiddenNeuronCount }, DataType.Float, 0);
            Parameter w2 = new Parameter(new int[] { this.dataset.OutputSize, hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter b2 = new Parameter(new int[] { this.dataset.OutputSize }, DataType.Float, 0);
            y = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w2, CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w1, x), b))), b2));
        }

        public void Train()
        {

            //ki kell egészíteni a gráfot
            Variable yt = Variable.InputVariable(new int[] {this.dataset.OutputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            //mivel tanítunk
            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            for (int epochI = 1; epochI <= epochCount; epochI++)
            {
                //Console.WriteLine($"Epoch {i} started.");

                double sumLoss = 0;
                double sumEval = 0;
                int samplesSeen = 0;

                //Console.WriteLine($"Count: {dataset.Count}");
                //Console.WriteLine($"InputSize: {dataset.InputSize}");
                //Console.WriteLine($"OutputSize: {dataset.OutputSize}");
                //Console.WriteLine($"Input values: {dataset.Input.Count}");
                //Console.WriteLine($"Output values: {dataset.Output.Count}");

                dataset.Shuffle();
                for (int start = 0; start < dataset.Count; start += batchSize)
                {
                    int currentBatchSize = Math.Min(batchSize, dataset.Count - start);

                    int inputStart = start * dataset.InputSize;
                    int inputCount = currentBatchSize * dataset.InputSize;

                    int outputStart = start * dataset.OutputSize;
                    int outputCount = currentBatchSize * dataset.OutputSize;

                    Value x_val = Value.CreateBatch(x.Shape, dataset.Input.GetRange(inputStart, inputCount), device);
                    Value yt_val = Value.CreateBatch(yt.Shape, dataset.Output.GetRange(outputStart, outputCount), device);

                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        {x, x_val },
                        {yt, yt_val }
                    };

                    bool isSweepEnd = start + currentBatchSize >= dataset.Count;

                    trainer.TrainMinibatch(inputDataMap, isSweepEnd, device);

                    int samples = (int)trainer.PreviousMinibatchSampleCount();

                    sumLoss += trainer.PreviousMinibatchLossAverage() * samples;
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * samples;

                    samplesSeen += samples;

                    //if (start < 1)
                    //{   
                    //    Console.WriteLine(
                    //        $"batch samples: {trainer.PreviousMinibatchSampleCount()}"
                    //    );

                    //    Console.WriteLine(
                    //        $"batch loss: {trainer.PreviousMinibatchLossAverage()}"
                    //    );

                    //    Console.WriteLine(
                    //        $"batch eval: {trainer.PreviousMinibatchEvaluationAverage()}"
                    //    );
                    //}
                }

                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", epochI, sumLoss / samplesSeen, sumEval / samplesSeen));
            }
        }

        //
        //public float Prediction(float age, float height, float weight, float salary)
        //{
        //    var inputDataMap = new Dictionary<Variable, Value>() { { x, LoadInput(age, height, weight, salary) } };
        //    var outputDataMap = new Dictionary<Variable, Value>() { { y, null } };

        //    y.Evaluate(inputDataMap, outputDataMap, device);

        //    return outputDataMap[y].GetDenseData<float>(y)[0][0];
        //}
    }
}
