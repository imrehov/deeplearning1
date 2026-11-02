using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace cntk_01;

class Program
{
    static void Main(string[] args)
    {
        Variable r = Variable.InputVariable(new int[] { 1 }, DataType.Float, "r");
        
        Constant two = Constant.Scalar(2.0f, DeviceDescriptor.CPUDevice);
        Constant pi = Constant.Scalar(Math.PI, DeviceDescriptor.CPUDevice);

        Function twor = CNTKLib.ElementTimes(r, two);
        Function perimeter = CNTKLib.ElementTimes(twor, pi);
        
        Function rsquared = CNTKLib.Square(r);
        Function area = CNTKLib.ElementTimes(rsquared, pi);


        Value r_value =Value.CreateBatch(new int[] {1}, new List<float>() {3.0f}, DeviceDescriptor.CPUDevice);

        var inputDataMap = new Dictionary<Variable, Value>()
        {
          {r, r_value},  
        };

        var outputDataMap = new Dictionary<Variable, Value>()
        {
            {perimeter, null},
        };

        perimeter.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);

        float perimeter_float = outputDataMap[perimeter].GetDenseData<float>(perimeter)[0][0];

        System.Console.WriteLine(perimeter_float);

    }
}
