(* ::Package:: *)

SetDirectory[NotebookDirectory[]];

(*Import
   columns: T, dt=0.01, dt=0.05, dt=0.1, dt=0.2, dt=0.4, dt=0.8 (fidelity values)
   no-trotter file columns: T, fidelity (long format, just two columns)*)

dTValues = {0.01, 0.05, 0.1, 0.2, 0.4, 0.8};

importTrotter[file_] := Module[{raw, header, dat, T, headerNumeric, colFor},
   raw = Import[file, "CSV"];
   header = First[raw];
   dat = Rest[raw];
   T = ToExpression /@ dat[[All, 1]];
   headerNumeric = Table[
     Quiet[ToExpression[StringReplace[header[[i]], "dt=" -> ""]]],
     {i, Length[header]}
     ];
   colFor[dtVal_] := Module[{colIdx, fidCol},
     colIdx = First[FirstPosition[headerNumeric, x_ /; NumericQ[x] && Abs[x - dtVal] < 10^-9]];
     fidCol = ToExpression /@ dat[[All, colIdx]];
     SortBy[Transpose[{T, 1 - fidCol}], First]
     ];
   Association[(# -> colFor[#]) & /@ dTValues]
   ];

trotterFile14 = "numerical_values/lmax_14_wmax_53/fidelity_vs_T_trotter.csv";
exactFile14 = "numerical_values/lmax_14_wmax_53/fidelity_vs_T_exact.csv";
trotterFile20 = "numerical_values/lmax_20_wmax_53/fidelity_vs_T_trotter.csv";
exactFile20 = "numerical_values/lmax_20_wmax_53/fidelity_vs_T_exact.csv";

Do[If[! FileExistsQ[f], Print["Missing: ", f]],
  {f, {trotterFile14, exactFile14, trotterFile20, exactFile20}}];

trotter14 = importTrotterWide[trotterFile14]; (* dTValue -> {T, infidelity} curve *)
exact14 = importTrotterWide[exactFile14]; (* dTValue -> {T, infidelity} curve *)
trotter20 = importTrotterWide[trotterFile20];
exact20 = importTrotterWide[exactFile20];

curves14 = trotter14;
curves20 = trotter20;

Print["--- Sanity check: infidelity at T=100 ---"];
Print["lmax=1.4 (trotter): ", Association[(# -> Cases[curves14[#], {100., inf_} :> inf]) & /@ dTValues]];
Print["lmax=1.4 (exact):   ", Association[(# -> Cases[exact14[#], {100., inf_} :> inf]) & /@ dTValues]];
Print["lmax=2.0 (trotter): ", Association[(# -> Cases[curves20[#], {100., inf_} :> inf]) & /@ dTValues]];
Print["lmax=2.0 (exact):   ", Association[(# -> Cases[exact20[#], {100., inf_} :> inf]) & /@ dTValues]];

dTColors = <|
   0.01 -> RGBColor[0.122, 0.467, 0.706],  (* blue *)
   0.05 -> RGBColor[1.000, 0.498, 0.055],  (* orange *)
   0.1 -> RGBColor[0.173, 0.627, 0.173],   (* green *)
   0.2 -> RGBColor[0.839, 0.153, 0.157],   (* red *)
   0.4 -> RGBColor[0.580, 0.404, 0.741],   (* purple *)
   0.8 -> RGBColor[0.549, 0.337, 0.294]    (* brown *)
   |>;

(* Panel sizing *)
panelWidth = 480;
plotAreaHeight = 340;
sideLabelSpace = 65; 
bottomLabelSpace = 55;
topLabelSpace = 10;    
plotAspectRatio = plotAreaHeight/panelWidth;

(* Build log-scale y-ticks *)
allInfidelities = Flatten[Join[
    Values[curves14][[All, All, 2]], Values[curves20][[All, All, 2]],
    Values[exact14][[All, All, 2]], Values[exact20][[All, All, 2]]
    ]];
allInfidelities = Select[allInfidelities, # > 0 &]; (* log undefined at/below 0 *)
logMinLabel = -4; (* lowest tick that actually gets a number *)
minorStep = Log10[2]; (* same spacing as a normal minor tick (the "2x" step) *)
logMin = logMinLabel - minorStep; 
logMax = Ceiling[Log10[Max[allInfidelities]]];
logMaxAxis = logMax + minorStep; 

majorPowers = Range[logMinLabel, logMax];
logYTicksLabeled = ({#, Superscript[10, #], {0.014, 0}} &) /@ majorPowers;
logYTicksBlank = ({#, "", {0.014, 0}} &) /@ majorPowers;
minorMultipliers = {2, 5};
logYMinor = Flatten[Table[
    {p + Log10[m], "", {0.007, 0}}, {p, logMinLabel, logMax - 1}, {m, minorMultipliers}], 1];
extraBottomTick = {logMin, "", {0.007, 0}}; (* single tick, one minor-step below 10^-4 *)
extraTopTick = {logMaxAxis, "", {0.007, 0}}; (* single tick, one minor-step above 10^0 *)
logYTicksLabeledFull = Join[logYTicksLabeled, logYMinor, {extraBottomTick, extraTopTick}];
logYTicksBlankFull = Join[logYTicksBlank, logYMinor, {extraBottomTick, extraTopTick}];

legendLabel[dt_] := Row[{
   Style["\[Delta]", FontFamily -> "Times New Roman"],
   Style["T", Italic, FontFamily -> "Times New Roman"],
   " = ", dt, " ",
   Superscript[Style["t", Italic, FontFamily -> "Times New Roman"], "-1"]
   }];
legendStyles = Join[
   (Directive[dTColors[#], AbsoluteThickness[4]] &) /@ dTValues,
   {Directive[Gray, AbsoluteThickness[3]], Directive[Gray, Dashed, AbsoluteThickness[3]]}
   ];
legendLabels = Join[legendLabel /@ dTValues,
   {Style["Trotter", FontFamily -> "Arial"], Style["Exact", FontFamily -> "Arial"]}];
legendGraphics = LineLegend[legendStyles, legendLabels,
   LabelStyle -> Directive[15, Black, FontFamily -> "Arial"],
   LegendMarkerSize -> {30, 18},
   Spacings -> {0, 0.6}
   ];
legendOffset = 29;
legendPositioned = Grid[{{Spacer[{1, legendOffset}]}, {legendGraphics}},
   Spacings -> {0, 0}, Alignment -> Left];

panelTitle[maxVal_] := Graphics[Text[
    Row[{
      Subscript[
       Style["\[Lambda]", 20, Black, FontFamily -> "Times New Roman"],
       Style["max", 14, Black, FontFamily -> "Times New Roman"]
       ],
      Style["=" <> maxVal, 20, Black, FontFamily -> "Arial"]
      }]
    ],
   ImageSize -> {panelWidth, 34}, PlotRange -> All,
   PlotRangePadding -> Scaled[0.2], ImagePadding -> {{35, 35}, {8, 8}}];

xMin = 0; 
xMax = Max[Flatten[{curves14[0.01][[All, 1]], curves20[0.01][[All, 1]]}]];
xTickVals = {xMin, 40, 80, 120, 160, xMax};
xMinorVals = Select[Range[Ceiling[xMin, 10], Floor[xMax, 10], 10],
   ! MemberQ[xTickVals, #] &]; 

xTicksLeftPanel = Join[
   ({#, Which[# == xMin, ToString[Round[xMin]], # == xMax, "", True, ToString[Round[#]]], {0.014, 0}} &) /@ xTickVals,
   ({#, "", {0.007, 0}} &) /@ xMinorVals
   ];
xTicksRightPanel = Join[
   ({#, Which[# == xMin, "", True, ToString[Round[#]]], {0.014, 0}} &) /@ xTickVals,
   ({#, "", {0.007, 0}} &) /@ xMinorVals
   ];
xTicksBlank = Join[
   ({#, "", {0.014, 0}} &) /@ xTickVals,
   ({#, "", {0.007, 0}} &) /@ xMinorVals
   ];

buildPanel[curves_, exactCurves_, isLeftPanel_] := Module[{plotLines, exactLines, logCurve},
   logCurve[data_] := {#[[1]], If[#[[2]] > 0, Log10[#[[2]]], logMin - 10]} & /@ data;
   plotLines = KeyValueMap[
     ListPlot[{logCurve[#2]}, Joined -> True,
       PlotStyle -> dTColors[#1],
       PlotMarkers -> {Automatic, 8}] &,
     curves
     ];

   exactLines = KeyValueMap[
     ListPlot[{logCurve[#2]}, Joined -> True,
       PlotStyle -> Directive[dTColors[#1], Dashed, Thickness[0.003]],
       PlotMarkers -> None] &,
     exactCurves
     ];
   Show[plotLines, exactLines,
    PlotRange -> {{xMin, xMax}, {logMin, logMaxAxis}},
    PlotRangePadding -> 0, (* the actual cause of the persistent gap at the
       boundaries -- PlotRange alone doesn't stop Mathematica from adding its
       own automatic padding around the specified range *)
    AspectRatio -> plotAspectRatio,
    ImageSize -> panelWidth,
    ImagePadding -> {
      {If[isLeftPanel, sideLabelSpace, 0], If[isLeftPanel, 0, sideLabelSpace]},
      {bottomLabelSpace, topLabelSpace}
      },
    Frame -> True,
    FrameStyle -> Directive[Black, Thickness[0.0025]],
    FrameTicksStyle -> Directive[15, FontFamily -> "Arial"],
    LabelStyle -> Directive[15, Black, FontFamily -> "Times New Roman"],
    FrameLabel -> {
      {If[isLeftPanel, 
   Row[{
     Style["1 - ", 18, Black, FontFamily -> "Times New Roman"], 
     Style["F", 18, Black, Italic, FontFamily -> "Times New Roman"]
   }], 
   None], 
 None},
      {Row[{Style["T", 18, Black, Italic, FontFamily -> "Times New Roman"],
         Style[" / ", 18, Black, FontFamily -> "Times New Roman"],
         Superscript[Style["t", 18, Black, Italic, FontFamily -> "Times New Roman"], "-1"]}], None}
      },
    FrameTicks -> {
      {If[isLeftPanel, logYTicksLabeledFull, logYTicksBlankFull], logYTicksBlankFull},
      {If[isLeftPanel, xTicksLeftPanel, xTicksRightPanel], xTicksBlank}
      },
    PlotRangeClipping -> True, 
    GridLines -> None 
    ]
   ];

panel14 = buildPanel[curves14, exact14, True];
panel20 = buildPanel[curves20, exact20, False];

titleRow = Grid[{{panelTitle["1.4"], panelTitle["2.0"]}},
   Spacings -> {0, 0}];
panelPair = Grid[{{panel14, panel20}}, Spacings -> {0, 0}, Alignment -> Top];

combined = Grid[{
    {titleRow, ""},
    {panelPair, legendPositioned}
    },
   Spacings -> {0, 0}, Alignment -> {Left, Top}]

(* --- Export --- *)s
Export["numerical_values/trotter_comparison_lmax14_vs_20_2.pdf", combined, ImageSize -> 900];
Export["numerical_values/trotter_comparison_lmax14_vs_20_2.png", combined,
  ImageSize -> 1400, ImageResolution -> 300, Background -> White];

