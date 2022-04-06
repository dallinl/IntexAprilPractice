using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace IntexApril.Models
{
    public class AccidentInfo
    {
        public float MILE_POINT { get; set; }
        public float WORK_ZONE_RELATED_True { get; set; }
        public float SINGLE_VEHICLE_True { get; set; }
        public float NIGHT_DARK_CONDITION_True { get; set; }
        public float CITY_MURRAY { get; set; }
        public float COUNTY_NAME_DAVIS { get; set; }
        public float COUNTY_NAME_WEBER { get; set; }
        public float MAIN_ROAD_NAME_Other { get; set; }

        public Tensor<float> AsTensor()
        {
            float[] data = new float[]
            {
                MILE_POINT, WORK_ZONE_RELATED_True, SINGLE_VEHICLE_True, NIGHT_DARK_CONDITION_True,
                CITY_MURRAY, COUNTY_NAME_DAVIS, COUNTY_NAME_WEBER, MAIN_ROAD_NAME_Other
            };
            int[] dimensions = new int[] { 1, 8 };
            return new DenseTensor<float>(data, dimensions);
        }

    }
}
