using CNTK;

namespace beadando
{
    internal class Program
    {
        static void Main(string[] args)
        {
            DeviceDescriptor.TrySetDefaultDevice(DeviceDescriptor.CPUDevice);

            DataSet traindata = new DataSet(@"..\..\..\..\train.csv");
            DataSet validationdata = new DataSet(@"..\..\..\..\validate.csv");

            BatchTraining app = new BatchTraining(traindata, validationdata, 3);

            app.TrainAndValidate();
            Console.ReadLine();
        }
    }
}
