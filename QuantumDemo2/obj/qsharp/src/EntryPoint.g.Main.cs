//------------------------------------------------------------------------------
// <auto-generated>                                                             
//     This code was generated by a tool.                                       
//     Changes to this file may cause incorrect behavior and will be lost if    
//     the code is regenerated.                                                 
// </auto-generated>                                                            
//------------------------------------------------------------------------------
using System;
using Microsoft.Quantum.Core;
using Microsoft.Quantum.Intrinsic;
using Microsoft.Quantum.Intrinsic.Interfaces;
using Microsoft.Quantum.Simulation.Core;

namespace __QsEntryPoint__
{
    internal class __QsEntryPoint__
    {
        private static async System.Threading.Tasks.Task<int> Main(string[] args) => await new global::Microsoft.Quantum.EntryPointDriver.Driver(new global::Microsoft.Quantum.EntryPointDriver.DriverSettings(simulatorOptionAliases: System.Collections.Immutable.ImmutableList.Create("--simulator", "-s"), quantumSimulatorName: "QuantumSimulator", sparseSimulatorName: "SparseSimulator", toffoliSimulatorName: "ToffoliSimulator", defaultSimulatorName: "QuantumSimulator", defaultExecutionTarget: "Any", targetCapability: "FullComputation", createDefaultCustomSimulator: () => throw new InvalidOperationException()), new global::Microsoft.Quantum.EntryPointDriver.IEntryPoint[] { new QuantumDemo2.__QsEntryPoint__QuantumSuperposition() }).Run(args);
    }
}