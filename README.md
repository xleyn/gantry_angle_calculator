<h1>The Gantry Angle Calculator Project</h1>
<br>
<p>Welcome to the Gantry Angle Calculator Project! Developed for the Regional Radiation Protection Service as part of my Professional Training Year, this is a tool to automate aspects of CT imaging unit testing.</p>
<h2>Background</h2>
<p>Part of the CT unit testing process is quality testing, namely evaluating whether the unit is functioning as required. During a scan, the gantry (body) of the CT scanner may be titled up to 30&deg off-axis for clinical reasons. The angle of rotation is determined by the scanner although one cannot assume that the true rotation is as expected - the agreement between the expected and true rotation is one aspect of the CT scanner that must be evaluated during Radiation Protection testing procedures.
<p>The testing procedure is as follows:
<ul>
  <li>Prepare a rectangle of Gafchromic Film, a material which darkens with exposure to radiation.</li>
  <li>Place the film in the isocentre of the CT scanner (the geometric centre of the gantry).</li>
  <li>Expose the film with radiation with a gantry angle of 0&deg (no rotation). This leads to a black band running down the centre of the film.</li>
  <li>Rotate the gantry to +30&deg and expose, leading to another black band.</li>
  <li>Rotate the gantry to -30&deg and expose, leading to a final black band.</li>
  <li>This leads to the following characteristic pattern on the film...</li>
  <img width="276" height="342" alt="image" src="https://github.com/user-attachments/assets/9ded633e-bb33-4f2d-8824-e043857be340" />
  <li>The true rotation angles for the nominal &plusmn30&deg rotations are measured manually using ImageJ plugin angles tool. The 0&deg exposure is used as a reference point i.e. angles are measured manually with respect to this.</li>
</ul>
</p>
<p>This method is flawed for multiple reasons, however the most significant factor is that repeat measurements (or measurements from different people) often yields widely different results! Due to tight acceptance criteria, this variation can even cause tests to pass or fail with repeat measurements. The inconsistency highlighted the need for a new solution with improved accuracy and repeatability - and this repository is the result! The Gantry Angle Calculator Project is a Python tool that utilises image processing methods to calculate the appropriate angles straight from a scanned image of the Gafchromic Film sample.</p>
<h2>Overview of Technical Implementation</h2>
<p>Here is a brief overview of the algorithm that was designed to achieve this:
<ul>
  <li>The image is cropped and rotated to be axis-aligned i.e. the edges of the film are parallel to the edges of the image.</li>
  <li>The edge of the film is located and an inset contour is drawn (see below)</li>
  <img width="299" height="403" alt="image" src="https://github.com/user-attachments/assets/46b136d6-4e3e-412d-bd86-6b2b1cb824d8" />
  <li>Walk along that contour, plotting a smoothed line profile of the pixel value (see below)</li>
  <img width="1611" height="846" alt="image" src="https://github.com/user-attachments/assets/51d92dd8-a3a7-4a3c-8460-ff04399335d5" />
  <li>The six visible peaks correspond to the line profile crossing each of the irradiated line profiles twice. From this, the positions for these 12 "crossing points" can be determined (see blue spots below)</li>
  <img width="300" height="402" alt="image" src="https://github.com/user-attachments/assets/f430a002-c559-412e-a7d0-5cf2263b4533" />
  <li>These points are then paired up as appropriate, to form lines. The required angles are then extracted by calculating the angles between lines using trigonometry. The angles are visually marked on an image of the film and this is both saved and displayed to the user at runtime. Finally, the numerical results are stored in a spreadsheet.</li>
<img width="585" height="574" alt="image" src="https://github.com/user-attachments/assets/24bf9247-95c1-47b9-ac5b-efe1c70aa4f1" />
</ul>
</p>
