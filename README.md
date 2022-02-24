# FS(Few-Spikes model Application)
`SNN` `EEG` `Depression detection` <br>
`A graduation project of an undergraduate from HUST` <br>
## Background
    Depression
      A global mental illness with unknown pathogenesis and heterogeneous in nature → Achieving accurate and early diagnosis 
      is challenging and extremely rewarding
    
    The detection of depression
      It's difficult for medicine to give a detailed and accurate explanation of the cause. It‘s mainly based on symptomology, 
    according to whether there are symptoms of depression, and how many items are met to make a diagnosis.
    
    Challenges in depression detection
      1. Diagnosis lacks biological diagnostic 'gold standard': Need to combine the patient's current mental state 
        and past history
      2. Diagnosis methods mostly limited to scales, which cannot effectively judge the subtype and severity of depression
      3. It has high requirements for clinicians, mainly relies on subjective evaluation, and has the possibility of false 
      positive diagnosis. Statistics show that general practitioners can only correctly identify 47.3% of patients, and there 
      is a large proportion of missed and misdiagnosed patients.

`The detetion based on deep learning:` <br>
```diff
  1. Modelable: Auxiliary Diagnosis of Depression → Quantitative Depression Status Assessment
  2. More acurrate: Eliminate the influence of subjective factors of doctors and patients in the diagnosis process, and 
  improve the accuracy of diagnosis
  3. Explainable: Depression cues are extracted from speech, facial expressions, fMRI images, EEG signals, eye movements, etc.
```

## Dataset
`20 healthy individuals and 20 depressed patients, 128-channel continuous EEG signals in resting state, sampling rate 250Hz` <br>

    Preprocessing:
    Filter → remove distractions(EMG、EOG, etc.) → Re-interpolate the removed bad derivatives → re-reference 
    → subspace reconstruction：Remove high energy components → Calculate inter-channel connection strength

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes'>
  <td width=276 valign=top style='width:207.25pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:Ayuthaya'>rhythm<o:p></o:p></span></p>
  </td>
  <td width=276 valign=top style='width:207.25pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:Ayuthaya'>band<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td width=276 valign=top style='width:207.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:"Cambria",serif;
  mso-bidi-font-family:Cambria'>δ</span><span lang=EN-US style='font-size:12.0pt;
  font-family:Ayuthaya'> rhythm<o:p></o:p></span></p>
  </td>
  <td width=276 valign=top style='width:207.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:Ayuthaya'>1-3
  Hz<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td width=276 valign=top style='width:207.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:"Cambria",serif;
  mso-bidi-font-family:Cambria'>θ</span><span lang=EN-US style='font-size:12.0pt;
  font-family:Ayuthaya'> rhythm<o:p></o:p></span></p>
  </td>
  <td width=276 valign=top style='width:207.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:Ayuthaya'>4-7
  Hz<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3'>
  <td width=276 valign=top style='width:207.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:"Cambria",serif;
  mso-bidi-font-family:Cambria'>α</span><span lang=EN-US style='font-size:12.0pt;
  font-family:Ayuthaya'> rhythm<o:p></o:p></span></p>
  </td>
  <td width=276 valign=top style='width:207.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:Ayuthaya'>8-13
  Hz<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4'>
  <td width=276 valign=top style='width:207.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:"Cambria",serif;
  mso-bidi-font-family:Cambria'>β</span><span lang=EN-US style='font-size:12.0pt;
  font-family:Ayuthaya'> rhythm<o:p></o:p></span></p>
  </td>
  <td width=276 valign=top style='width:207.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:Ayuthaya'>14-30
  Hz<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;mso-yfti-lastrow:yes'>
  <td width=276 valign=top style='width:207.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:"Cambria",serif;
  mso-bidi-font-family:Cambria'>γ</span><span lang=EN-US style='font-size:12.0pt;
  font-family:Ayuthaya'> rhythm<o:p></o:p></span></p>
  </td>
  <td width=276 valign=top style='width:207.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:Ayuthaya'>40-100
  Hz<o:p></o:p></span></p>
  </td>
 </tr>
</table>

    Calculating the PLV(Phase Locking Vector):
    PLV measures the connectivity between two channels([0,1])
    The closer it is to 1, the stronger the synchronization between the two regions.
    The closer it is to 0, the stronger the mutual independence of the two regions.
    Then get 128x128 data of EEG
    
## Modeling

### Designed Convolutional Networks
    5 layers of Convolutional Networks, CNN: see CNN5.py CNN5_2
    
        Train from scratch：
          set the parameter use_pretrained_model = False
        Use pretrained model:
          load /checkpoints/CNN5_2_3.pth
          the test accuracy achieves 99.6% |the train accuracy and val accuracy achieve 89.3% and 88.5%

******
### Few-Spikes model
