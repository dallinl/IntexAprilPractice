using IntexApril.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace IntexApril.Controllers
{
    public class HomeController : Controller
    {
        private InferenceSession _session;
        //private AccidentInfo _context { get; set; }

        public HomeController(InferenceSession session)
        {
            _session = session;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }
        
        public IActionResult SummaryPage()
        {
            return View();
        }

        [HttpGet]
        public IActionResult PredictionCalculator()
        {
            return View();
        }

        [HttpPost]
        public IActionResult PredictionCalculator(AccidentInfo data)
        {
            var result = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("float_input", data.AsTensor())
            });
            Tensor<float> score = result.First().AsTensor<float>();
            var prediction = new CrashPrediction { CrashSeverity = score.First()};
            //result.Dispose();
            return View("PredictionResult", prediction);
        }

    }
}
