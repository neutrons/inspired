# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'help_oclimax.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Help_OCLIMAX(object):
    def setupUi(self, Help_OCLIMAX):
        Help_OCLIMAX.setObjectName("Help_OCLIMAX")
        Help_OCLIMAX.resize(640, 480)
        self.verticalLayout = QtWidgets.QVBoxLayout(Help_OCLIMAX)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Help_OCLIMAX)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.textBrowser = QtWidgets.QTextBrowser(Help_OCLIMAX)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)

        self.retranslateUi(Help_OCLIMAX)
        QtCore.QMetaObject.connectSlotsByName(Help_OCLIMAX)

    def retranslateUi(self, Help_OCLIMAX):
        _translate = QtCore.QCoreApplication.translate
        Help_OCLIMAX.setWindowTitle(_translate("Help_OCLIMAX", "Dialog"))
        self.label.setText(_translate("Help_OCLIMAX", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">OCLIMAX Help</span></p></body></html>"))
        self.textBrowser.setHtml(_translate("Help_OCLIMAX", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:\'Segoe UI\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">This is a GUI for OCLIMAX [6], with enhanced features for single crystal simulations.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">For Phonopy calculation, a gamma-centered Q mesh will be used to sample the Brillouin zone. If mesh.conf exists in the working directory, the default mesh will be loaded. Otherwise, it can be specified and saved. Ideally, the Q mesh should be sufficiently fine to achieve converged INS simulation (at a specific bin size and resolution) without creating a mesh.yaml file that is too large to handle.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">BAND_POINTS scaling factor is used to increase sampling along the dispersion curve. The default value is usually good. If you see spotty/disconnected dispersion spectra in a single crystal simulation, you may try to increase this number.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">For a complete reference to the OCLIMAX parameters, please check oclimax_manual.pdf in the oclimax folder or online at https://sites.google.com/site/ornliceman/download. Note that &quot;Task&quot; on the GUI is redefined to combine &quot;TASK&quot; and &quot;INSTR&quot; in the original params file for simplicity/clarity. The options are:</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; text-decoration: underline;\">1D VISION/TOSCA inc</span><span style=\" font-size:12pt;\">: 1D spectra as measured on VISION/TOSCA calculated using incoherent approximation</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; text-decoration: underline;\">1D VISION/TOSCA full</span><span style=\" font-size:12pt;\">: 1D spectra as measured on VISION/TOSCA with full calculation (including coherent scattering effects)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; text-decoration: underline;\">2D S(Q,E) powder inc</span><span style=\" font-size:12pt;\">: 2D spectra as measured on direct geometry spectrometers, powder sample under incoherent approximation</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; text-decoration: underline;\">2D S(Q,E) powder full</span><span style=\" font-size:12pt;\">: 2D spectra as measured on direct geometry spectrometers, powder sample with full calculation (including coherent scattering effects)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; text-decoration: underline;\">2D S(Q,E) single crystal</span><span style=\" font-size:12pt;\">: 2D [</span><span style=\" font-size:12pt; font-weight:700;\">Q</span><span style=\" font-size:12pt;\">,E] spectra as measured on direct geometry spectrometers, single crystal coherent scattering</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; text-decoration: underline;\">2D S(Q,Q) single crystal</span><span style=\" font-size:12pt;\">: 2D [</span><span style=\" font-size:12pt; font-weight:700;\">Qx</span><span style=\" font-size:12pt;\">,</span><span style=\" font-size:12pt; font-weight:700;\">Qy</span><span style=\" font-size:12pt;\">] spectra as measured on direct geometry spectrometers, single crystal coherent scattering</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; text-decoration: underline;\">Others (defined in params file)</span><span style=\" font-size:12pt;\">: Other less common options can be edited manually in the params file. See oclimax_manual for more information.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Additional parameters to streamline single crystal simulations are provided on the right panel, which are explained below.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Q1, Q2, and Q3 are the three unit vectors in the reciprocal space.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">x-axis can be Q1, Q2, or Q3, and y-axis can be E for S(Q,E) simulations (5th opition in Task), or any other Q for S(Qx,Qy) simulations (6th option in Task).</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Q_bin and E_bin should be set according to the Task and x/y-axis. For example, if Q1 is chosen for x-axis, then the three numbers in Q1_bin define x-axis as [Qmin, Qstep, Qmax]. If E is chosen for y-axis, the three numbers in E_bin define y-axis as [Emin, Estep, Emax]. In this case, Q2 and Q3 are not associated with any axis, and they are to be (numerically) integrated over the specified ranges. The integration can also be described by three numbers [Qmin, Qstep, Qmax], where Qmin and Qmax define the range of integration, and Qstep is the sampling interval that determines how many samples should be calculated in this range.  For example, the default values [-0.05, 0.05, 0.05] mean that Q2 and Q3 will be integrated within -0.05 and +0.05, with three sampling slices calculated at -0.05, 0.0, +0.05 each (thus nine samples in total for the combination of Q2 and Q3).</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Parameters can be either edited in the GUI, or loaded from a *.params file. The modified parameters in the GUI should be saved to a *.params file to take effect. This GUI only provides access to the most commonly used parameters. Additional parameters not in the GUI will have to be changed directly in the *.params file before running the INS simulation. The default file name is oclimax.params, but it is recommended to save the file in a unique name (with keywords describing the simulation) so that it is not accidentally overwritten and you can quickly reproduce the simulation in the future.</span></p></body></html>"))