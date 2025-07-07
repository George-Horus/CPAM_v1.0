"""
#########################################################################
Author: <Zhang YH>
Vegetation Multispectral Indices Database (VMID)
This document defines a class for calculating vegetation index.
Can be applied to the fields of agricultural remote sensing monitoring.
It can be used to quickly calculate 117 the vegetation index and derive it.
Each function will indicate the source paper.
Welcome to keep adding.
#########################################################################
"""
import numpy as np
import pandas as pd

class VMID:
    """
    Vegetation Index Library

    This module calculates 117 published multispectral vegetation indices and exports the results.

    1. Contains 117 published multispectral vegetation indices.

    2. Inputs include five parameters: red, green, blue, red_edge, and nir.

    3. Each band parameter is optional. If a specific band is missing, indices requiring that band are skipped.

    4. The parameter `index_list` is also optional. For details, see the `_calculate_indices` function.

    5. When the input is a one-dimensional array, the output is a DataFrame, where each column represents a vegetation index.

    6. When the input is a two-dimensional or multi-dimensional array, the function returns a dictionary of arrays, where vegetation index names can be used as keys to retrieve the corresponding multi-dimensional index arrays.

    7. The vegetation index calculation functions include references for each index, formatted as follows:
       Author surnames and initials. Article title in sentence style. Journal Title. Year; volume(issue): page range. DOI
    """
    def __init__(self, red=None, green=None, blue=None, red_edge=None, nir=None, index_list=None):
        """
        Parameters
        ----------
        blue : float, optional
            Primary wavelength of visible blue light (nm) (default = 480.)
        green : float, optional
            Primary wavelength of visible green light (nm) (default = 550.)
        red : float, optional
            Primary wavelength of visible red light (nm) (default = 670.)
        nir : float, optional
            Primary wavelength of near-infrared light (nm) (default = 800.)
        red_edge : float, optional
            Primary wavelength of near-infrared light (nm) (default = 717.)
        """

        # Store input arrays or None if not provided
        self.bands = {'R': red, 'G': green, 'B': blue, 'RE': red_edge, 'NIR': nir}

        self.all_indices = ['ARI', 'ARVI', 'BDRVI', 'CCCI', 'CIG', 'CIRE', 'CIVE','CRI700', 'CVI', 'DATT', 'DVI','EBVI',
                            'EGVI', 'ERVI', 'EVI1', 'EVI2', 'ExG', 'ExGR', 'EXR1', 'EXR2', 'GARI', 'GDVI1','GDVI2', 'GEMI',
                            'GLI1', 'GLI2', 'GNDVI', 'GOSAVI', 'GRDVI', 'GRVI1', 'GRVI2', 'GSAVI','GWDRVI', 'MCARI', 'MCARI1',
                            'MCARI2', 'MCARI3', 'MCARI4', 'MDD', 'MEVI', 'MGRVI', 'MNDI1', 'MNDI2', 'MNDRE1', 'MNDRE2', 'MNDVI',
                            'MNLI', 'MNSI', 'MRETVI', 'MSAVI', 'MSR', 'MSR_G', 'MSR_R', 'MSR_RE', 'MSRRE', 'MSRREDRE', 'MTCARI',
                            'MTCI', 'NB', 'NDGI', 'NDRE', 'NDVI', 'NDWI', 'NG', 'NGBDI', 'NGI', 'NGRDI', 'NNIR', 'NNRI', 'NPCI',
                            'NR', 'NREI1', 'NREI2', 'NRI', 'NROV', 'OSAVI1', 'OSAVI2', 'PNDVI', 'PPR', 'PRI', 'PSRI1','PSRI2',
                            'PVR', 'RBI', 'RBNDVI', 'RDVI1', 'RDVI2', 'REDVI', 'REGDVI', 'REGNDVI', 'REGRVI', 'RENDVI', 'REOSAVI',
                            'RERVI', 'RESAVI', 'RESR', 'RETVI', 'REWDRVI', 'RGBVI', 'RGI', 'RI', 'RTVIcore', 'RVI', 'SAVI', 'SIPI1',
                            'SIPI2', 'SRI', 'SRPI', 'TCARI', 'TVI1', 'TVI2', 'VARI', 'VI700', 'VIopt', 'WBI', 'WDRVI', 'WI']

        # Calculate and store each index
        self.indices = {}
        self._calculate_indices(index_list)

    def export_all(self):
        """
        Outputs the results of vegetation index calculations.
        If the input has two or more dimensions, returns a dictionary of arrays.
        If the input has fewer than two dimensions, returns a DataFrame.
        """
        # 输出指数计算结果，当维度大于等于2时，返回数组字典，小于2时返回dataframe
        # Output the index calculation result, and return the array dictionary when the dimension is greater than or equal to 2, and return the dataframe when it is less than 2.
        for band_name, band_data in self.bands.items():
            # 如果当前波段数据不为 None
            if band_data is not None:
                if band_data.ndim >= 2:  # 检查波段数据的维度
                    return self.indices
                else:
                    df = pd.DataFrame(self.indices)
                    return df
        # 如果所有波段数据都为 None，可以抛出一个异常或返回 None
        raise ValueError("All band data are None.")

    def _calculate_indices(self,index_list=None):
        # Dynamically calculate indices only when required bands are available
        """
        Calculates vegetation indices based on the provided index_list:

        - If index_list is None, calculates all indices.
        - If index_list is an empty list [], calculates only NDVI.
        - Otherwise, calculates only the specified indices.

        :param index_list: List of vegetation index names, e.g. ['ARVI', 'BDRVI']
        :return: Dictionary containing the calculation results for the selected indices.
        """

        if index_list is None:
            index_list = self.all_indices
        elif len(index_list) == 0:
            index_list = ['NDVI']

        if 'BLUE' in index_list and  self.bands['B'] is not None:
            self.indices['BLUE'] = self._BLUE()

        if 'GREEN' in index_list and  self.bands['G'] is not None:
            self.indices['GREEN'] = self._GREEN()

        if 'RED' in index_list and  self.bands['R'] is not None:
            self.indices['RED'] = self._RED()

        if 'RedEdge' in index_list and  self.bands['RE'] is not None:
            self.indices['RedEdge'] = self._RedEdge()

        if 'NIR' in index_list and  self.bands['NIR'] is not None:
            self.indices['NIR'] = self._NIR()

        if 'ARI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['ARI'] = self._ARI()

        if 'ARVI' in index_list and self.bands['R'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['ARVI'] = self._ARVI()

        if 'BDRVI' in index_list and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['BDRVI'] = self._BDRVI()

        if 'CCCI' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['CCCI'] = self._CCCI()

        if 'CIG' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['CIG'] = self._CIG()

        if 'CIRE' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['CIRE'] = self._CIRE()

        if 'CIVE' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['CIVE'] = self._CIVE()

        if 'CRI700' in index_list and self.bands['B'] is not None and self.bands['RE'] is not None:
            self.indices['CRI700'] = self._CRI700()

        if 'CVI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['CVI'] = self._CVI()

        if 'DATT' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['DATT'] = self._DATT()

        if 'DVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['DVI'] = self._DVI()

        if 'EBVI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['EBVI'] = self._EBVI()

        if 'EGVI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['EGVI'] = self._EGVI()

        if 'ERVI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['ERVI'] = self._ERVI()

        if 'EVI1' in index_list and self.bands['R'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['EVI1'] = self._EVI1()

        if 'EVI2' in index_list and self.bands['R'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['EVI2'] = self._EVI2()

        if 'ExG' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['ExG'] = self._ExG()

        if 'ExGR' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['ExGR'] = self._ExGR()

        if 'EXR1' in index_list and self.bands['R'] is not None and self.bands['B'] is not None:
            self.indices['EXR1'] = self._EXR1()

        if 'EXR2' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['EXR2'] = self._EXR2()

        if 'GARI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['GARI'] = self._GARI()

        if 'GDVI1' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['GDVI1'] = self._GDVI1()

        if 'GDVI2' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['GDVI2'] = self._GDVI2()

        if 'GEMI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['GEMI'] = self._GEMI()

        if 'GLI1' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['GLI1'] = self._GLI1()

        if 'GLI2' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['GLI2'] = self._GLI2()

        if 'GNDVI' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['GNDVI'] = self._GNDVI()

        if 'GOSAVI' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['GOSAVI'] = self._GOSAVI()

        if 'GRDVI' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['GRDVI'] = self._GRDVI()

        if 'GRVI1' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['GRVI1'] = self._GRVI1()

        if 'GRVI2' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['GRVI2'] = self._GRVI2()

        if 'GSAVI' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['GSAVI'] = self._GSAVI()

        if 'GWDRVI' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['GWDRVI'] = self._GWDRVI()

        if 'MCARI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['MCARI'] = self._MCARI()

        if 'MCARI1' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MCARI1'] = self._MCARI1()

        if 'MCARI2' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MCARI2'] = self._MCARI2()

        if 'MCARI3' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MCARI3'] = self._MCARI3()

        if 'MCARI4' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MCARI4'] = self._MCARI4()

        if 'MDD' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MDD'] = self._MDD()

        if 'MEVI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MEVI'] = self._MEVI()

        if 'MGRVI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['MGRVI'] = self._MGRVI()

        if 'MNDI1' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MNDI1'] = self._MNDI1()

        if 'MNDI2' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MNDI2'] = self._MNDI2()

        if 'MNDRE1' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MNDRE1'] = self._MNDRE1()

        if 'MNDRE2' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MNDRE2'] = self._MNDRE2()

        if 'MNDVI' in index_list and self.bands['R'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['MNDVI'] = self._MNDVI()

        if 'MNLI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['MNLI'] = self._MNLI()

        if 'MNSI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MNSI'] = self._MNSI()

        if 'MRETVI' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MRETVI'] = self._MRETVI()

        if 'MSAVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['MSAVI'] = self._MSAVI()

        if 'MSR' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['MSR'] = self._MSR()

        if 'MSR_G' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['MSR_G'] = self._MSR_G()

        if 'MSR_R' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['MSR_R'] = self._MSR_R()

        if 'MSR_RE' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MSR_RE'] = self._MSR_RE()

        if 'MSRRE' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MSRRE'] = self._MSRRE()

        if 'MSRREDRE' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MSRREDRE'] = self._MSRREDRE()

        if 'MTCARI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MTCARI'] = self._MTCARI()

        if 'MTCI' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['MTCI'] = self._MTCI()

        if 'NB' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['NB'] = self._NB()

        if 'NDGI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['NDGI'] = self._NDGI()

        if 'NDRE' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['NDRE'] = self._NDRE()

        if 'NDVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['NDVI'] = self._NDVI()

        if 'NDWI' in index_list and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['NDWI'] = self._NDWI()

        if 'NG' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['NG'] = self._NG()

        if 'NGBDI' in index_list and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['NGBDI'] = self._NGBDI()

        if 'NGI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['NGI'] = self._NGI()

        if 'NGRDI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['NGRDI'] = self._NGRDI()

        if 'NNIR' in index_list and self.bands['NIR'] is not None and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['NNIR'] = self._NNIR()

        if 'NNRI' in index_list and self.bands['B'] is not None and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['NNRI'] = self._NNRI()

        if 'NPCI' in index_list and self.bands['B'] is not None and self.bands['R'] is not None:
            self.indices['NPCI'] = self._NPCI()

        if 'NR' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['NR'] = self._NR()

        if 'NREI1' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None and self.bands['G'] is not None:
            self.indices['NREI1'] = self._NREI1()

        if 'NREI2' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['NREI2'] = self._NREI2()

        if 'NRI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['NRI'] = self._NRI()

        if 'NROV' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['NROV'] = self._NROV()

        if 'OSAVI1' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['OSAVI1'] = self._OSAVI1()

        if 'OSAVI2' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['OSAVI2'] = self._OSAVI2()

        if 'PNDVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['PNDVI'] = self._PNDVI()

        if 'PPR' in index_list and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['PPR'] = self._PPR()

        if 'PRI' in index_list and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['PRI'] = self._PRI()

        if 'PSRI1' in index_list and self.bands['R'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['PSRI1'] = self._PSRI1()

        if 'PSRI2' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['PSRI2'] = self._PSRI2()

        if 'PVR' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['PVR'] = self._PVR()

        if 'RBI' in index_list and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['RBI'] = self._RBI()

        if 'RBNDVI' in index_list and self.bands['R'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['RBNDVI'] = self._RBNDVI()

        if 'RDVI1' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['RDVI1'] = self._RDVI1()

        if 'RDVI2' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['RDVI2'] = self._RDVI2()

        if 'REDVI' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['REDVI'] = self._REDVI()

        if 'REGDVI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['REGDVI'] = self._REGDVI()

        if 'REGNDVI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['REGNDVI'] = self._REGNDVI()

        if 'REGRVI' in index_list and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['REGRVI'] = self._REGRVI()

        if 'RENDVI' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['RENDVI'] = self._RENDVI()

        if 'REOSAVI' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['REOSAVI'] = self._REOSAVI()

        if 'RERVI' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['RERVI'] = self._RERVI()

        if 'RESAVI' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None:
            self.indices['RESAVI'] = self._RESAVI()

        if 'RESR' in index_list and self.bands['RE'] is not None and self.bands['R'] is not None:
            self.indices['RESR'] = self._RESR()

        if 'RETVI' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None and self.bands['R'] is not None:
            self.indices['RETVI'] = self._RETVI()

        if 'REWDRVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['REWDRVI'] = self._REWDRVI()

        if 'RGBVI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['RGBVI'] = self._RGBVI()

        if 'RGI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['RGI'] = self._RGI()

        if 'RI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None:
            self.indices['RI'] = self._RI()

        if 'RTVIcore' in index_list and self.bands['RE'] is not None and self.bands['NIR'] is not None and self.bands['G'] is not None:
            self.indices['RTVIcore'] = self._RTVIcore()

        if 'RVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['RVI'] = self._RVI()

        if 'SAVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['SAVI'] = self._SAVI()

        if 'SIPI1' in index_list and self.bands['R'] is not None and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['SIPI1'] = self._SIPI1()

        if 'SIPI2' in index_list and self.bands['B'] is not None and self.bands['NIR'] is not None:
            self.indices['SIPI2'] = self._SIPI2()

        if 'SRI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['SRI'] = self._SRI()

        if 'SRPI' in index_list and self.bands['B'] is not None and self.bands['R'] is not None:
            self.indices['SRPI'] = self._SRPI()

        if 'TCARI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['RE'] is not None:
            self.indices['TCARI'] = self._TCARI()

        if 'TVI1' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['NIR'] is not None:
            self.indices['TVI1'] = self._TVI1()

        if 'TVI2' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['TVI2'] = self._TVI2()

        if 'VARI' in index_list and self.bands['G'] is not None and self.bands['R'] is not None and self.bands['B'] is not None:
            self.indices['VARI'] = self._VARI()

        if 'VI700' in index_list and self.bands['R'] is not None and self.bands['RE'] is not None:
            self.indices['VI700'] = self._VI700()

        if 'VIopt' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['VIopt'] = self._VIopt()

        if 'WBI' in index_list and self.bands['B'] is not None and self.bands['R'] is not None:
            self.indices['WBI'] = self._WBI()

        if 'WDRVI' in index_list and self.bands['R'] is not None and self.bands['NIR'] is not None:
            self.indices['WDRVI'] = self._WDRVI()

        if 'WI' in index_list and self.bands['R'] is not None and self.bands['G'] is not None and self.bands['B'] is not None:
            self.indices['WI'] = self._WI()

    # Multispectral indices formula
    def _BLUE(self):
        B = self.bands['B']
        return B

    def _GREEN(self):
        G = self.bands['G']
        return G

    def _RED(self):
        R = self.bands['R']
        return R

    def _RedEdge(self):
        RE = self.bands['RE']
        return RE

    def _NIR(self):
        NIR = self.bands['NIR']
        return NIR

    def _ARI(self):
        """
          Indices_name:
          Anthocyanin Reflectance Index  (ARI)
          article:
          Karnieli A, Kaufman YJ, Remer L,Wald A. AFRI - aerosol free vegetation index. Remote Sensing of Environment. 2001; 77: 10–21. DOI: 10.1016/S0034-4257(01)00190-0
        """
        G = self.bands['G']
        RE = self.bands['RE']
        return (1 / G) - (1 / RE)

    def _ARVI(self):
        """
          Indices_name:
          Atmospherically Resistant Vegetation Index (ARVI)
          article:
          Kaufman YJ, Tanre D. Atmospherically resistant vegetation index (ARVI) for EOS-MODIS. IEEE Transactions on Geoscience and Remote Sensing. 1992; 30(2): 261–270. DOI: 10.1109/36.134076
        """
        R = self.bands['R']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return (NIR - (2 * R - B)) / (NIR + (2 * B - R))

    def _BDRVI(self):
        """
          Indices_name:
          Blue-wide dynamic range vegetation index (BDRVI)
          article:
          Hancock DW, Dougherty CT. Relationships between blue-and red-based vegetation indices and leaf area and yield of alfalfa. Crop Science. 2007; 47(6): 2547–2556. DOI: 10.2135/cropsci2007.01.0031
        """
        B = self.bands['B']
        NIR = self.bands['NIR']
        return (0.1 * NIR - B) / (0.1 * NIR + B)

    def _CCCI(self):
        """
          Indices_name:
          Canopy Chlorophyll Content Index (CCCI)
          article:
          Fitzgerald G, Rodriguez D, O’Leary G. Measuring and predicting canopy nitrogen nutrition in wheat using a spectral index—the canopy chlorophyll content index (CCCI). Field Crops Research. 2010; 116(3): 318–324. DOI: 10.1016/j.fcr.2010.01.010
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return ((NIR - RE) / (NIR + RE)) / ((NIR - R) / (NIR + R))

    def _CIG(self):
        """
          Indices_name:
          Green Chlorophyll Index (CIG)
          article:
          Gitelson AA, Viña A, Ciganda V, Rundquist DC, Arkebauer TJ. Remote estimation of canopy chlorophyll content in crops: art. no. L08403. Geophysical Research Letters. 2005; 32(8): L08403. DOI: 10.1029/2005GL022688
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return NIR / G - 1

    def _CIRE(self):
        """
          Indices_name:
          Red Edge Chlorophyll Index (CIRE)
          article:
          Gitelson AA, Viña A, Ciganda V, Rundquist DC, Arkebauer TJ. Remote estimation of canopy chlorophyll content in crops: art. no. L08403. Geophysical Research Letters. 2005; 32(8): L08403. DOI: 10.1029/2005GL022688
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return NIR / RE - 1

    def _CIVE(self):
        """
          Indices_name:
          Color Index of Vegetation Extraction (CIVE)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return 0.441 * R - 0.881 * G + 0.3856 * B + 18.78745

    def _CRI700(self):
        """
          Indices_name:
          Carotenoid Reflectance Index 700  (CRI700)
          article:
          Gitelson A, Merzlyak MN, Chivkunova OB. Optical properties and nondestructive estimation of anthocyanin content in plant leaves. Photochemistry and Photobiology. 2001; 74: 38–45. DOI: 10.1562/0031-8655(2001)074<0038:opaneo>2.0.co;2
        """
        B = self.bands['B']
        RE = self.bands['RE']
        return (1 / B) - (1 / RE)

    def _CVI(self):
        """
          Indices_name:
          Chlorophyll Vegetation Index  (CVI)
          article:
          Hunt ER, Daughtry CST, Eitel JUH, Long DS. Remote sensing leaf chlorophyll content using a visible Band index. Agronomy Journal. 2011; 103: 1090–1099. DOI: 10.2134/agronj2010.0395
        """
        R = self.bands['R']
        G = self.bands['G']
        NIR = self.bands['NIR']
        return (NIR / G) * (R / G)

    def _DATT(self):
        """
          Indices_name:
          DATT Index (DATT)
          article:
          Datt B. Visible/near infrared reflectance and chlorophyll content in Eucalyptus leaves. International Journal of Remote Sensing. 1999; 20(14): 2741–2759. DOI: 10.1080/014311699211778
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (NIR - R)

    def _DVI(self):
        """
          Indices_name:
          Difference Vegetation Index (DVI)
          article:
          Tucker CJ. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sensing of Environment. 1979; 8(2): 127–150. DOI: 10.1016/0034-4257(79)90013-0
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return NIR - R

    def _EBVI(self):
        """
          Indices_name:
          Excess Blue Vegetation Index (EBVI)
          article:
          Meyer GE, Neto JC. Verification of color vegetation indices for automated crop imaging applications. Computers and Electronics in Agriculture. 2008; 63(2): 282–293. DOI: 10.1016/j.compag.2008.03.009
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return 1.4 * (B - G) / (R + B + G)

    def _EGVI(self):
        """
          Indices_name:
          Excess Green Vegetation Index (EGVI)
          article:
          Woebbecke DM, Meyer GE, VonBargen K, Mortensen DA. Color indexes for weed identification under various soil, residue, and lighting conditions. Transactions of the ASAE.1995;38(1): 259–269.
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return 2 * (G - R - B) / (R + G + B)

    def _ERVI(self):
        """
          Indices_name:
          Excess Red Vegetation Index (ERVI)
          article:
          Meyer GE, Neto JC. Verification of color vegetation indices for automated crop imaging applications. Computers and Electronics in Agriculture. 2008; 63(2): 282–293. DOI: 10.1016/j.compag.2008.03.009
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return 1.4 * R / (R + B + G) - G / (R + B + G)

    def _EVI1(self):
        """
          Indices_name:
          Enhanced Vegetation Index (EVI1)
          article:
          Huete A, Didan K, Miura T, Rodriguez EP, Gao X, Ferreira LG. Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sensing of Environment. 2002; 83(1-2): 195–213. DOI: 10.1016/S0034-4257(02)00096-2
        """
        R = self.bands['R']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1)

    def _EVI2(self):
        """
          Indices_name:
          Enhanced Vegetation Index-2  (EVI2)
          article:
          Jiang Z, Huete AR, Didan K, Miura T. Development of a two-band enhanced vegetation index without a blue band. Remote Sensing of Environment. 2008; 112(10): 3833–3845. DOI: 10.1016/j.rse.2008.06.006
        """
        R = self.bands['R']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return 2.5 * (NIR - R) / (NIR + 6 * R + 2.4 * B + 1)

    def _ExG(self):
        """
          Indices_name:
          Excess Green Index (ExG)
          article:
          Meyer GE, Neto JC. Verification of color vegetation indices for automated crop imaging applications. Computers and Electronics in Agriculture. 2008; 63(2): 282–293. DOI: 10.1016/j.compag.2008.03.009
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return 2 * G - R - B

    def _ExGR(self):
        """
          Indices_name:
          Excess Green Minus Red Index (ExGR)
          article:
          Tran KH, Zhang XY, Ketchpaw AR, Wang JM, Ye YC, Shen Y. A novel algorithm for the generation of gap-free time series by fusing harmonized Landsat 8 and Sentinel-2 observations with PhenoCam time series for detecting land surface phenology. Remote Sensing of Environment. 2022; 282: 113275. DOI: 10.1016/j.rse.2022.113275
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return 3 * G - 2.4 * R - B

    def _EXR1(self):
        """
          Indices_name:
          Excess Red Index (EXR1)
          article:
          Meyer GE, Hindman TW, Laksmi K. Machine vision detection parameters for plant species identification. Proceedings of SPIE – The International Society for Optical Engineering. 1999. DOI: 10.1117/12.336896
        """
        R = self.bands['R']
        B = self.bands['B']
        return 1.4 * R - B

    def _EXR2(self):
        """
          Indices_name:
          Excess Red Index (EXR2)
          article:
          Meyer GE, Neto JC. Verification of color vegetation indices for automated crop imaging applications. Computers and Electronics in Agriculture. 2008; 63(2): 282–293. DOI: 10.1016/j.compag.2008.03.009
        """
        R = self.bands['R']
        G = self.bands['G']
        return 1.4 * R - G

    def _GARI(self):
        """
          Indices_name:
          Green Atmospherically Resistant Vegetation Index (GARI)
          article:
          Gitelson AA, Kaufman YJ, Merzlyak MN. Use of a green channel in remote sensing of global vegetation from EOS-MODIS. Remote Sensing of Environment. 1996; 58(3): 289–298. DOI: 10.1016/s0034-4257(96)00072-7
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return NIR - (G - (B - R)) / NIR + (G - (B - R))


    def _GDVI1(self):
        """
          Indices_name:
          Green Difference Vegetation Index (GDVI1)
          article:
          Tucker CJ. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sensing of Environment. 1979; 8(2): 127–150. DOI: 10.1016/0034-4257(79)90013-0
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return NIR - G


    def _GDVI2(self):
        """
          Indices_name:
          Generalized Difference Vegetation Index (GDVI2)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return (NIR / R - 1) / (NIR / R + 1)


    def _GEMI(self):
        """
          Indices_name:
          Global Environment Monitoring Index (GEMI)
          article:
          Pinty B., Verstraete MM. GEMI: a non-linear index to monitor global vegetation from satellites. Vegetatio. 1992; 101: 15–20. DOI: 10.1007/BF00031911
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return (2 * (NIR ** 2 - R ** 2)+1.5 * NIR + 0.5 * R)/(NIR + R + 0.5)


    def _GLI1(self):
        """
          Indices_name:
          Green Leaf Index (GLI1)
          article:
          Louhaichi M, Borman MM, Johnson DE. Spatially located platform and aerial photography for documentation of grazing impacts on wheat. Geocarto International. 2001; 16(1): 65–70. DOI: 10.1080/10106040108542184
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return (2 * G - R - B) / (2 * G + R + B)


    def _GLI2(self):
        """
          Indices_name:
          Green Leaf Index 2 (GLI2)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return (2 * G - R + B) / (2 * G + R + B)


    def _GNDVI(self):
        """
          Indices_name:
          Green Normalized Difference Vegetation Index (GNDVI)
          article:
          Gitelson AA, Merzlyak MN. Remote sensing of chlorophyll concentration in higher plant leaves. Advances in Space Research. 1998; 22(5): 689–692. DOI: 10.1016/S0273-1177(97)01133-2
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return (NIR - G) / (NIR + G)


    def _GOSAVI(self):
        """
          Indices_name:
          Green Optimal Soil Adjusted Vegetation Index (GOSAVI)
          article:
          Rondeaux G, Steven M, Baret F. Optimization of soil-adjusted vegetation indices. Remote Sensing of Environment. 1996; 55(2): 95–107.
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return (1 + 0.16) * (NIR - G) / (NIR + G + 0.16)


    def _GRDVI(self):
        """
          Indices_name:
          Green Re-normalized Different Vegetation Index (GRDVI)
          article:
          Roujean JL, Breon FM. Estimating PAR absorbed by vegetation from bidirectional reflectance measurements. Remote Sensing of Environment. 1995; 51(3): 375–384. DOI: 10.1016/0034-4257(94)00114-3
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return (NIR - G) / np.sqrt(NIR + G)


    def _GRVI1(self):
        """
          Indices_name:
          Green-red vegetation index (GRVI1)
          article:
          Tucker CJ. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sensing of Environment. 1979; 8: 127–150. DOI: 10.1016/0034-4257(79)90013-0
        """
        R = self.bands['R']
        G = self.bands['G']
        return (G - R) / (G + R)


    def _GRVI2(self):
        """
          Indices_name:
          Green Ratio Vegetation Index (GRVI2)
          article:
          Buschmann C, Nagel E. In vivo spectroscopy and internal optics of leaves as basis for remote sensing of vegetation. International Journal of Remote Sensing. 1993; 14: 711–722.
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return NIR / G


    def _GSAVI(self):
        """
          Indices_name:
          Green Soil Adjusted Vegetation Index (GSAVI)
          article:
          Sripada RP, Heiniger RW, White JG, Meijer AD. Aerial color infrared photography for determining early in-season nitrogen requirements in corn. Agronomy Journal. 2006; 98: 968–977.
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return 1.5 * (NIR - G) / (NIR + G + 0.5)


    def _GWDRVI(self):
        """
          Indices_name:
          Green Wide Dynamic Range Vegetation Index (GWDRVI)
          article:
          Gitelson AA. Wide dynamic range vegetation index for remote quantification of biophysical characteristics of vegetation. Journal of Plant Physiology. 2004; 161: 165–173.
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return (0.12 * NIR - G) / (0.12 * NIR + G)


    def _MCARI(self):
        """
          Indices_name:
          Modified Chlorophyll Absorption in Reflectance Index (MCARI)
          article:
          Daughtry CST, Walthall CL, Kim MS, et al. Estimating corn leaf chlorophyll concentration from leaf and canopy reflectance. Remote Sensing of Environment. 2000; 74(2): 229–239.
        """
        R = self.bands['R']
        G = self.bands['G']
        RE = self.bands['RE']
        return ((RE - R) - 0.2 * (RE - G)) * (RE / R)


    def _MCARI1(self):
        """
          Indices_name:
          Modified Chlorophyll Absorption In Reflectance Index 1 (MCARI1)
          article:
          Gitelson AA. Wide dynamic range vegetation index for remote quantification of biophysical characteristics of vegetation. Journal of Plant Physiology. 2004; 161: 165–173.
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return ((NIR - RE) - 0.2 * (NIR - R)) * (NIR / RE)


    def _MCARI2(self):
        """
          Indices_name:
          Modified Chlorophyll Absorption In Reflectance Index 2 (MCARI2)
          article:
          Haboudane D, Miller JR, Pattey E, Zarco-Tejada PJ, Strachan IB. Hyperspectral vegetation indices and novel algorithms for predicting green LAI of crop canopies: Modeling and validation in the context of precision agriculture. Remote Sensing of Environment. 2004; 90(3): 337–352. DOI: 10.1016/j.rse.2003.12.013
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 1.5 * (2.5 * (NIR - R) - 1.3 * (NIR - RE)) / np.sqrt((2 * NIR + 1) ** 2 - (6 * NIR - 5 * np.sqrt(R) - 0.5))


    def _MCARI3(self):
        """
          Indices_name:
          Modified Chlorophyll Absorption In Reflectance Index 3 (MCARI3)
          article:
          Haboudane D, Miller JR, Pattey E, Zarco-Tejada PJ, Strachan IB. Hyperspectral vegetation indices and novel algorithms for predicting green LAI of crop canopies: Modeling and validation in the context of precision agriculture. Remote Sensing of Environment. 2004; 90(3): 337–352. DOI: 10.1016/j.rse.2003.12.013
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return ((NIR - RE) - 0.2 * (NIR - R)) * (NIR / RE)


    def _MCARI4(self):
        """
          Indices_name:
          Modified Chlorophyll Absorption In Reflectance Index 4 (MCARI4)
          article:
          Haboudane D, Miller JR, Pattey E, Zarco-Tejada PJ, Strachan IB. Hyperspectral vegetation indices and novel algorithms for predicting green LAI of crop canopies: Modeling and validation in the context of precision agriculture. Remote Sensing of Environment. 2004; 90(3): 337–352. DOI: 10.1016/j.rse.2003.12.013
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (1.5 * (2.5 * (NIR - G) - 1.3 * (NIR - RE)) / (np.sqrt((2 * NIR + 1) ** 2 - (6 * NIR - 5 * np.sqrt(G)) - 0.5)))


    def _MDD(self):
        """
          Indices_name:
          Modified Double Difference Index (MDD)
          article:
          Le Maire G, Francois C, Dufrene E. Towards universal broad leaf chlorophyll indices using PROSPECT simulated database and hyperspectral reflectance measurements. Remote Sensing of Environment. 2004; 89: 1–28.
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (RE - G)


    def _MEVI(self):
        """
          Indices_name:
          Modified enhanced vegetation index (MEVI)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 2.5 * (NIR - RE) / (NIR + 6 * RE - 7.5 * G + 1)


    def _MGRVI(self):
        """
          Indices_name:
          Modified Green Red Vegetation Index (MGRVI)
          article:
          Vincent L. Morphological grayscale reconstruction in image analysis: applications and efficient algorithms. IEEE Transactions on Image Processing. 1993; 2(2): 176–201. DOI: 10.1109/83.217222
        """
        R = self.bands['R']
        G = self.bands['G']
        return (G ** 2 - R ** 2) / (G ** 2 + R ** 2)


    def _MNDI1(self):
        """
          Indices_name:
          Modified Normalized Difference Index (MNDI1)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (NIR - G)


    def _MNDI2(self):
        """
          Indices_name:
          Modified Normalized Difference Index (MNDI2)
          article:
          Datt B. Visible/near infrared reflectance and chlorophyll content in Eucalyptus leaves. International Journal of Remote Sensing. 1999; 20: 2741–2759.
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (NIR + RE - 2 * R)


    def _MNDRE1(self):
        """
          Indices_name:
          Modified Normalized Difference Red Edge (MNDRE1)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE - 2 * G) / (NIR + RE - 2 * G)


    def _MNDRE2(self):
        """
          Indices_name:
          Modified Normalized Difference Red Edge (MNDRE2)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE + 2 * R) / (NIR + RE - 2 * R)


    def _MNDVI(self):
        """
          Indices_name:
          Modified Normalized Difference Vegetation Index      (MNDVI)
          article:
          Main R, Cho MA, Mathieu R, O'Kennedy MM, Ramoelo A, Koch S. An investigation into robust spectral indices for leaf chlorophyll estimation. ISPRS Journal of Photogrammetry and Remote Sensing. 2011; 66(6): 751–761. DOI: 10.1016/j.isprsjprs.2011.08.001
        """
        R = self.bands['R']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return (NIR - R) / (NIR + R - 2 * B)


    def _MNLI(self):
        """
          Indices_name:
          Modified Normalized Difference Leaf Index (MNLI)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        R = self.bands['R']
        G = self.bands['G']
        return (R - G) / (R + G)


    def _MNSI(self):
        """
          Indices_name:
          Misra Non-such Index  (MNSI)
          article:
          Misra PN, Wheeler SG, Oliver RE. Kauth-Thomas brightness and greenness axes. In: Contract NASA 9-14350, 23–46, 1977.
        """
        R = self.bands['R']
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 0.404 * G + 0.039 * R - 0.505 * RE + 0.762 * NIR


    def _MRETVI(self):
        """
          Indices_name:
          Modified Red Edge Transformed Vegetation Index (MRETVI)
          article:
          Haboudane D, Miller JR, Pattey E, Zarco-Tejada PJ, Strachan IB. Hyperspectral vegetation indices and novel algorithms for predicting green LAI of crop canopies: modeling and validation in the context of precision agriculture. Remote Sensing of Environment. 2004; 90: 337–352.
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 1.2 * (1.2 * (NIR - R) - 2.5 * (RE - R))


    def _MSAVI(self):
        """
          Indices_name:
          Modified Soil-Adjusted Vegetation Index (MSAVI)
          article:
          Qi J, Chehbouni A, Huete AR, Kerr YH, Sorooshian S. A modified soil adjusted vegetation index. Remote Sensing of Environment. 1994; 48(2): 119–126.
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return (2 * NIR + 1 - np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - R))) / 2


    def _MSR(self):
        """
          Indices_name:
          Modified Simple Radio (MSR)
          article:
          Gitelson AA, Gritz Y, Merzlyak MN. Relationships between leaf chlorophyll content and spectral reflectance and algorithms for non-destructive chlorophyll assessment in higher plant leaves. Journal of Plant Physiology. 2003; 160(3): 271–282. DOI: 10.1078/0176-1617-00887
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        # 避免除以0
        ratio = np.divide(NIR, R + 1e-6)
        # 计算表达式
        msr_raw = (ratio - 1) / (ratio + 1)
        # 避免 sqrt 负数
        msr = np.sqrt(np.clip(msr_raw, 0, None))
        return msr


    def _MSR_G(self):
        """
          Indices_name:
          Modified Green Simple Ratio (MSR_G)
          article:
          Chen JM. Evaluation of vegetation indices and a modified simple ratio for boreal applications. Canadian Journal of Remote Sensing. 1996; 22: 229–242.
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return (NIR / G - 1) / np.sqrt(NIR / G + 1)


    def _MSR_R(self):
        """
          Indices_name:
          Modified Red Simple Ratio (MSR_R)
          article:
          Chen JM, Cihlar J. Retrieving leaf area index of boreal conifer forests using Landsat TM images. Remote Sensing of Environment. 1996; 55(2): 153–162.
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return (NIR / R - 1) / np.sqrt(NIR / R + 1)


    def _MSR_RE(self):
        """
          Indices_name:
          Modified Red Edge Simple Ratio (MSR_RE)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR / RE - 1) / np.sqrt(NIR / RE + 1)


    def _MSRRE(self):
        """
          Indices_name:
          Modified Simple Ratio Red-edge  (MSRRE)
          article:
          Wu C, Niu Z, Tang Q, Huang W. Estimating chlorophyll content from hyperspectral vegetation indices: Modeling and validation. Agricultural and Forest Meteorology. 2008; 148(8-9): 1230–1241. DOI: 10.1016/j.agrformet.2008.03.005
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR / RE) - 1


    def _MSRREDRE(self):
        """
          Indices_name:
          Red and Red-edge MSR Index  (MSRREDRE)
          article:
          Xie Q, Dash J, Huang W, Peng D, Qin Q, Mortimer H, Dong Y. Vegetation indices combining the red and red-edge spectral information for leaf area index retrieval. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 2018; 11(5): 1482–1493. DOI: 10.1109/JSTARS.2018.2813281
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR / (0.2 * R + (1 - 0.2) * RE)) - 1

    def _MTCARI(self):
        """
          Indices_name:
          Modified transformed CARI (MTCARI)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 3 * ((NIR - RE) - 0.2 * (NIR - G) * (NIR / RE))


    def _MTCI(self):
        """
          Indices_name:
          MERIS Terrestrial Chlorophyll Index (MTCI)
          article:
          Dash J, Curran PJ. MTCI: The MERIS terrestrial chlorophyll index. Geoscience and Remote Sensing Symposium, 2004. IGARSS ’04. Proceedings. 2004 IEEE International. 2004; 151–161. DOI: 10.1109/IGARSS.2004.1369009
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (RE - R)


    def _NB(self):
        """
          Indices_name:
          Normalized Blue (NB)
          article:
          Saberioon M, Amin M, Anuar A, Gholizadeh A, Wayayok A, Khairunniza-Bejo S. Assessment of rice leaf chlorophyll content using visible bands at different growth stages at both the leaf and canopy scale. International Journal of Applied Earth Observation and Geoinformation. 2014; 32: 35–45. DOI: 10.1016/j.jag.2014.03.018
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return B / (R + G + B)


    def _NDGI(self):
        """
          Indices_name:
          Normalized Difference Green Degree Index (NDGI)
          article:
          Lyon JG, Yuan D, Lunetta RS. A change detection experiment using vegetation indices. Photogrammetric Engineering & Remote Sensing. 1998; 64(2): 143–150.
        """
        R = self.bands['R']
        G = self.bands['G']
        return (G - R) / (G + R)


    def _NDRE(self):
        """
          Indices_name:
          Normalized Difference Red Edge (NDRE)
          article:
          Sims DA, Gamon JA. Relationships between leaf pigment content and spectral reflectance across a wide range of species, leaf structures and developmental stages. Remote Sensing of Environment. 2002; 81(2-3): 337–354. DOI: 10.1016/S0034-4257(02)00010-X
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (NIR + RE)


    def _NDVI(self):
        """
          Indices_name:
          Normalized Difference Vegetation Index (NDVI)
          article:
          Rouse JW, Haas RH, Deering DW. Monitoring the vernal advancement and retrogradation (green wave effect) of natural vegetation. Goddard Space Flight Center. 1973.
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return (NIR - R) / (NIR + R)


    def _NDWI(self):
        """
          Indices_name:
          Normalized Difference Water Index  (NDWI)
          article:
          McFeeters SK. The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing. 1996; 17(7): 1425–1432. DOI: 10.1080/01431169608948714
        """
        G = self.bands['G']
        NIR = self.bands['NIR']
        return (G - NIR) / (G + NIR)


    def _NG(self):
        """
          Indices_name:
          Normalized Green (NG)
          article:
          Saberioon M, Amin M, Anuar A, Gholizadeh A, Wayayok A, Khairunniza-Bejo S. Assessment of rice leaf chlorophyll content using visible bands at different growth stages at both the leaf and canopy scale. International Journal of Applied Earth Observation and Geoinformation. 2014; 32: 35–45. DOI: 10.1016/j.jag.2014.03.018
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return G / (R + G + B)


    def _NGBDI(self):
        """
          Indices_name:
          Normalized Green - Blue Difference Index (NGBDI)
          article:
          Verrelst J, Schaepman ME, Koetz B, et al. Angular sensitivity analysis of vegetation indices derived from CHRIS/PROBA data. Remote Sensing of Environment. 2008; 112(5): 2341–2353.
        """
        G = self.bands['G']
        B = self.bands['B']
        return (G - B) / (G + B)


    def _NGI(self):
        """
          Indices_name:
          Normalized Green Index (NGI)
          article:
          Sripada RP, Heiniger RW, White JG, Meijer AD. Aerial color infrared photography for determining early in-season nitrogen requirements in corn. Agronomy Journal. 2006; 98: 968–977.
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return G / (NIR + G + RE)


    def _NGRDI(self):
        """
          Indices_name:
          Normalized Green Red Difference Index  (NGRDI)
          article:
          Zarco-Tejada PJ, Miller JR, Noland TL, Mohammed GH, Sampson PH. Scaling-up and model inversion methods with narrowband optical indices for chlorophyll content estimation in closed forest canopies with hyperspectral data. IEEE Transactions on Geoscience and Remote Sensing. 2001; 39: 1491–1507. DOI: 10.1109/36.934080
        """
        G = self.bands['G']
        RE = self.bands['RE']
        return (G - RE) / (G + RE)


    def _NNIR(self):
        """
          Indices_name:
          Normalized NIR Index (NNIR)
          article:
          Sripada RP, Heiniger RW, White JG, Meijer AD. Aerial color infrared photography for determining early in-season nitrogen requirements in corn. Agronomy Journal. 2006; 98: 968–977.
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return NIR / (NIR + G + RE)


    def _NNRI(self):
        """
          Indices_name:
          Normalized NIR Redness Index (NNRI)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        B = self.bands['B']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - 2 * RE + B) / (NIR + 2 * RE + B)


    def _NPCI(self):
        """
          Indices_name:
          Chlorophyll Normalized Vegetation Index (NPCI)
          article:
          Clay DE, Kim KI, Chang J, et al. Characterizing water and nitrogen stress in corn using remote sensing. Agronomy Journal. 2006; 98(3): 579–587.
        """
        R = self.bands['R']
        B = self.bands['B']
        return (R - B) / (R + B)


    def _NR(self):
        """
          Indices_name:
          Normalized Red (NR)
          article:
          Saberioon M, Amin M, Anuar A, Gholizadeh A, Wayayok A, Khairunniza-Bejo S. Assessment of rice leaf chlorophyll content using visible bands at different growth stages at both the leaf and canopy scale. International Journal of Applied Earth Observation and Geoinformation. 2014; 32: 35–45. DOI: 10.1016/j.jag.2014.03.018
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return R / (R + G + B)


    def _NREI1(self):
        """
          Indices_name:
          Normalized Red Edge Index (NREI1)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return RE / (NIR + G + RE)


    def _NREI2(self):
        """
          Indices_name:
          Normalized Red Edge Index (NREI2)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (NIR + RE)


    def _NRI(self):
        """
          Indices_name:
          Normalized Red Index (NRI)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        R = self.bands['R']
        G = self.bands['G']
        return (G - R) / (G + R)


    def _NROV(self):
        """
          Indices_name:
          Near-infrared Reflectance of Vegetation (NROV)
          article:
          Badgley G, Field CB, Berry JA. Canopy near-infrared reflectance and terrestrial photosynthesis. Science Advances. 2017; 3(3): e1602244. DOI: 10.1126/sciadv.1602244
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        NDVI = (NIR-R)/(NIR+R)
        return NDVI * NIR


    def _OSAVI1(self):
        """
          Indices_name:
          Optimized Soil-Adjusted Vegetation Index (OSAVI1)
          article:
          Rondeaux G, Steven M, Baret F. Optimization of soil-adjusted vegetation indices. Remote Sensing of Environment. 1996; 55: 95–107. DOI: 10.1016/0034-4257(95)00186-7
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return 0.16 * (NIR - R) / (NIR + R + 0.16)


    def _OSAVI2(self):
        """
          Indices_name:
          Optimized Soil-Adjusted Vegetation Index (OSAVI2)
          article:
          Rondeaux G, Steven M, Baret F. Optimization of soil-adjusted vegetation indices. Remote Sensing of Environment. 1996; 55: 95–107. DOI: 10.1016/0034-4257(95)00186-7
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return 1.16 * NIR - (R / NIR) + R + 0.16


    def _PNDVI(self):
        """
          Indices_name:
          Projected Normalized Difference Vegetation Index (PNDVI)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        # 避免除以0
        denominator = NIR + R + 1e-6
        numerator = NIR - R
        ratio = numerator / denominator
        # 避免 sqrt 负值
        pndvi = np.sqrt(np.clip(ratio, 0, None))
        return pndvi


    def _PPR(self):
        """
          Indices_name:
          Plant Pigment Ratio  (PPR)
          article:
          Metternich G. Vegetation indices derived from high-resolution airborne videography for precision crop management. International Journal of Remote Sensing. 2003; 24(14): 2855–2877. DOI: 10.1080/01431160210163074
        """
        G = self.bands['G']
        B = self.bands['B']
        return (G - B) / (G + B)


    def _PRI(self):
        """
          Indices_name:
          Photochemical Reflectance Index (PRI)
          article:
          Gammon JA, Peninsulas J, Field CB. A narrow-waveband spectral index that tracks diurnal changes in photosynthetic efficiency. Remote Sensing of Environment. 1992; 41(1): 35–44. DOI: 10.1016/0034-4257(92)90059-S
        """
        G = self.bands['G']
        B = self.bands['B']
        return (B - G) / (B + G)


    def _PSRI1(self):
        """
          Indices_name:
          Plant Senescence Reflectance Index (PSRI1)
          article:
          Merzlyak MN, Gitelson AA, Chivkunova OB, Rakitin VY. Non‐destructive optical detection of pigment changes during leaf senescence and fruit ripening. Physiologia Plantarum. 1999; 106(1): 135–141. DOI: 10.1034/j.1399-3054.1999.106119.x
        """
        R = self.bands['R']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return (R - B) / NIR


    def _PSRI2(self):
        """
          Indices_name:
          Plant Senescence Reflectance Index  (PSRI2)
          article:
          Merzlyak MN, Gitelson AA, Chivkunova OB, Rakitin VY. Non‐destructive optical detection of pigment changes during leaf senescence and fruit ripening. Physiologia Plantarum. 1999; 106(1): 135–141. DOI: 10.1034/j.1399-3054.1999.106119.x
        """
        R = self.bands['R']
        G = self.bands['G']
        RE = self.bands['RE']
        return (R - G) / RE


    def _PVR(self):
        """
          Indices_name:
          Photosyntetic Vigour Ratio  (PVR)
          article:
          Metternicht G. Vegetation indices derived from high-resolution airborne videography for precision crop management. International Journal of Remote Sensing. 2003; 24(14): 2855–2877. DOI: 10.1080/01431160210163074
        """
        R = self.bands['R']
        G = self.bands['G']
        return (G - R) / (G + R)


    def _RBI(self):
        """
          Indices_name:
          Ratio blue index (RBI)
          article:
          Pearson RL, Miller LD. Remote mapping of standing crop biomass for estimation of productivity of the shortgrass prairie. Remote Sensing of Environment. 1972; VIII. DOI: 10.1177/002076409904500102
        """
        B = self.bands['B']
        NIR = self.bands['NIR']
        return NIR / B


    def _RBNDVI(self):
        """
          Indices_name:
          Red-Blue NDVI  (RBNDVI)
          article:
          Wang FM, Huang JF, Tang YL, Wang XZ. New vegetation index and its application in estimating leaf area index of rice. Rice Science. 2007; 14(3): 195–203. DOI: 10.1016/S1672-6308(07)60027-4
        """
        R = self.bands['R']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return (NIR - (R + B)) / (NIR + (R + B))


    def _RDVI1(self):
        """
          Indices_name:
          Renormalized Difference Vegetation Index (RDVI1)
          article:
          Roujean JL, Breon FM. Estimating PAR absorbed by vegetation from bidirectional reflectance measurements. Remote Sensing of Environment. 1995; 51(3): 375–384. DOI: 10.1016/0034-4257(94)00114-3
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        denominator = np.sqrt(np.clip(NIR + R, 1e-6, None))  # 避免负数或0
        rdvi1 = (NIR - R) / denominator
        return rdvi1


    def _RDVI2(self):
        """
          Indices_name:
          Renormalized Difference Vegetation Index (RDVI2)
          article:
          Roujean JL, Breon FM. Estimating PAR absorbed by vegetation from bidirectional reflectance measurements. Remote Sensing of Environment. 1995; 51(3): 375–384. DOI: 10.1016/0034-4257(94)00114-3
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        NDVI = (NIR - R) / (NIR + R + 1e-6)  # 加 epsilon 防止除以0
        DVI = NIR - R
        value = NDVI + DVI
        value = np.where(value < 0, 0, value)  # 裁剪负值避免 sqrt 警告
        return np.sqrt(value)


    def _REDVI(self):
        """
          Indices_name:
          Red Edge Difference Vegetation Index (REDVI)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return NIR - RE


    def _REGDVI(self):
        """
          Indices_name:
          Red edge green difference vegetation index (REGDVI)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        G = self.bands['G']
        RE = self.bands['RE']
        return RE - G


    def _REGNDVI(self):
        """
          Indices_name:
          Red edge GNDVI (REGNDVI)
          article:
          Gitelson AA, Kaufman YJ, Merzlyak MN. Use of a green channel in remote sensing of global vegetation from EOS-MODIS. Remote Sensing of Environment. 1996; 58: 289–298.
        """
        G = self.bands['G']
        RE = self.bands['RE']
        return (RE - G) / (RE + G)


    def _REGRVI(self):
        """
          Indices_name:
          Red edge green ratio vegetation index (REGRVI)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        G = self.bands['G']
        RE = self.bands['RE']
        return RE / G


    def _RENDVI(self):
        """
          Indices_name:
          Red Edge Normalized Difference Vegetation Index (RENDVI)
          article:
          Daughtry CST, Gallo KP, Goward SN, et al. Spectral estimates of absorbed radiation and phytomass production in corn and soybean canopies. Remote Sensing of Environment. 1992; 39(2): 141–152.
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return (NIR - RE) / (NIR + RE)


    def _REOSAVI(self):
        """
          Indices_name:
          Red Edge Optimal Soil Adjusted Vegetation Index (REOSAVI)
          article:
          Rondeaux G, Steven M, Baret F. Optimization of soil-adjusted vegetation indices. Remote Sensing of Environment. 1996; 55: 95–107.
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 1.5 * (NIR - RE) / (NIR + RE + 0.5)


    def _RERVI(self):
        """
          Indices_name:
          Red Edge Ratio Vegetation Index (RERVI)
          article:
          Gitelson AA, Kaufman YJ, Merzlyak MN. Use of a green channel in remote sensing of global vegetation from EOS-MODIS. Remote Sensing of Environment. 1996; 58: 289–298.
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return NIR / RE


    def _RESAVI(self):
        """
          Indices_name:
          Red edge soil adjusted vegetation index (RESAVI)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 1.5 * ((NIR - RE) / (NIR + RE + 0.5))


    def _RESR(self):
        """
          Indices_name:
          Red Edge Simple Ratio (RESR)
          article:
          Erdle K, Mistele B, Schmidhalter U. Comparison of active and passive spectral sensors in discriminating biomass parameters and nitrogen status in wheat cultivars. Field Crops Research. 2011; 124(1): 74–84. DOI: 10.1016/j.fcr.2011.06.007
        """
        R = self.bands['R']
        RE = self.bands['RE']
        return RE / R


    def _RETVI(self):
        """
          Indices_name:
          Red Edge Transformed Vegetation Index (RETVI)
          article:
          Broge NH, Leblanc E. Comparing prediction power and stability of broadband and hyperspectral vegetation indices for estimation of green leaf area index and canopy chlorophyll density. Remote Sensing of Environment. 2000; 76: 156–172.
        """
        R = self.bands['R']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 0.5 * (120 * (NIR - R) - 200 * (RE - R))


    def _REWDRVI(self):
        """
          Indices_name:
          Red Edge Wide Dynamic Range Vegetation Index (REWDRVI)
          article:
          Cao Q, Miao Y, Wang H, Huang S, Cheng S, Khosla R, Jiang R. Non-destructive estimation of rice plant nitrogen status with Crop Circle multispectral active canopy sensor. Field Crops Research. 2013; 154: 133–144. DOI: 10.1016/j.fcr.2013.08.005
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return (0.12 * NIR - R) / (0.12 * NIR + R)


    def _RGBVI(self):
        """
          Indices_name:
          Red Green Blue Vegetation Index (RGBVI)
          article:
          Aasen H, Kirchgessner N, Walter A, Liebisch F. PhenoCams for field phenotyping: Using very high temporal resolution digital repeated photography to investigate interactions of growth, phenology, and harvest traits. Frontiers in Plant Science. 2020; 11: 593.
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return (G ** 2 - B * R) / (G ** 2 + B * R)


    def _RGI(self):
        """
          Indices_name:
          Red-Green Index (RGI)
          article:
          Anees A, Aryal J. Near-Real Time Detection of Beetle Infestation in Pine Forests Using MODIS Data. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 2014; 7(9): 3713–3723. DOI: 10.1109/JSTARS.2014.2330830
        """
        R = self.bands['R']
        G = self.bands['G']
        return R / G


    def _RI(self):
        """
          Indices_name:
          Redness Index (RI)
          article:
          Huete AR, Escadafal R. Assessment of biophysical soil properties through spectral decomposition techniques. Remote Sensing of Environment. 1991; 35(2/3): 149–159.
        """
        R = self.bands['R']
        G = self.bands['G']
        return (R - G) / (R + G)


    def _RTVIcore(self):
        """
          Indices_name:
          Red-edge Triangular Vegetation Index  (RTVIcore)
          article:
          Chen PF, Nicolas T, Wang JH, Philippe V, Huang WJ, Li BG. New index for crop canopy fresh biomass estimation. Spectroscopy and Spectral Analysis. 2010; 30(2): 512–517. DOI: 10.3964/j.issn.1000-0593(2010)02-0512-06
        """
        G = self.bands['G']
        RE = self.bands['RE']
        NIR = self.bands['NIR']
        return 100 * (NIR - RE) - 10 * (NIR - G)


    def _RVI(self):
        """
          Indices_name:
          Ratio Vegetation Index (RVI)
          article:
          Pearson RL, Miller LD. Remote mapping of standing crop biomass for estimation of the productivity of the shortgrass prairie. Proceedings of the Eighth International Symposium on Remote Sensing of Environment. 1972; 1355.
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return NIR / R


    def _SAVI(self):
        """
          Indices_name:
          Soil-Adjusted Vegetation Index (SAVI)
          article:
          Tucker CJ. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sensing of Environment. 1979; 8: 127–150.
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return 1.5 * (NIR - R) / (NIR + R + 0.5)


    def _SIPI1(self):
        """
          Indices_name:
          Structure Insensitive Pigment Index (SIPI1)
          article:
          Penuelas J, Baret F, Filella I. Semi-empirical indices to assess carotenoids/chlorophyll a ratio from leaf spectral reflectance. Photosynthetica. 1995; 31(2): 221–230.
        """
        R = self.bands['R']
        B = self.bands['B']
        NIR = self.bands['NIR']
        return (NIR - B) / (NIR - R)


    def _SIPI2(self):
        """
          Indices_name:
          Structure-Independent Pigment Index (SIPI2)
          article:
          Penuelas J, Baret F, Filella I. Semi-empirical indices to assess carotenoids/chlorophyll a ratio from leaf spectral reflectance. Photosynthetica. 1995; 31(2): 221–230.
        """
        B = self.bands['B']
        NIR = self.bands['NIR']
        return (NIR - B) / (NIR + B)


    def _SRI(self):
        """
          Indices_name:
          Simple Ratio Index  (SRI)
          article:
          Jordan CF. Derivation of leaf-area index from quality of light on the forest floor. Ecology. 1969; 50: 663–666. DOI: 10.2307/1936256
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return NIR / R


    def _SRPI(self):
        """
          Indices_name:
          Simple Ratio Pigment Index (SRPI)
          article:
          Chappelle EW, Kim MS, McMurtrey JE III. Ratio analysis of reflectance spectra (RARS): An algorithm for the remote estimation of the concentrations of chlorophyll A, chlorophyll B, and carotenoids in soybean leaves. Remote Sensing of Environment. 1992; 39(3): 239–247.
        """
        R = self.bands['R']
        B = self.bands['B']
        return B / R


    def _TCARI(self):
        """
          Indices_name:
          Transformed Chlorophyll Absorption in Reflectance Index (TCARI )
          article:
          Devadas R, Lamb D, Simpfendorfer S, Backhouse D. Evaluating ten spectral vegetation indices for identifying rust infection in individual wheat leaves. Precision Agriculture. 2009; 10: 459–470. DOI: 10.1007/s11119-008-9100-2
        """
        R = self.bands['R']
        G = self.bands['G']
        RE = self.bands['RE']
        return 3 * (((RE - R) - 0.2 * (RE - G)) * (RE / R))


    def _TVI1(self):
        """
          Indices_name:
          Triangular Vegetation Index (TVI1)
          article:
          Tomas AD, Nieto H, Guzinski R, et al. Multi-scale approach of the surface temperature/vegetation index triangle method for estimating evapotranspiration over heterogeneous landscapes. EGU General Assembly. 2012.
        """
        R = self.bands['R']
        G = self.bands['G']
        NIR = self.bands['NIR']
        tvi1 = 60 * (NIR - G) - 100 * (G - R)
        tvi1 = np.where(np.isfinite(tvi1), tvi1, np.nan)  # 把 inf / -inf 替换为 nan
        return tvi1


    def _TVI2(self):
        """
          Indices_name:
          Transformed Vegetation Index (TVI2)
          article:
          Broge NH, Leblanc E. Comparing prediction power and stability of broadband and hyperspectral vegetation indices for estimation of green leaf area index and canopy chlorophyll density. Remote Sensing of Environment. 2001; 76(2): 156–172. DOI: 10.1016/S0034-4257(00)00197-8
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        NDVI = (NIR-R)/(NIR+R)
        NDVI_safe = np.clip(NDVI + 0.5, 0, None)  # 保证 sqrt 中的值不小于 0
        return np.sqrt(NDVI_safe)


    def _VARI(self):
        """
          Indices_name:
          Visible Atmospherically Resistant Index (VARI)
          article:
          Gitelson AA, Kaufman YJ, Stark R, Rundquist D. Novel algorithms for remote estimation of vegetation fraction. Remote Sensing of Environment. 2002; 80(1): 76–87. DOI: 10.1016/S0034-4257(01)00289-9
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return (G - R) / (G + R - B)


    def _VI700(self):
        """
          Indices_name:
          Vegetation Index 700  (VI700)
          article:
          Gitelson AA, Kaufman YJ, Stark R, Rundquist D. Novel algorithms for remote estimation of vegetation fraction. Remote Sensing of Environment. 2002; 80(1): 76–87. DOI: 10.1016/S0034-4257(01)00289-9
        """
        R = self.bands['R']
        RE = self.bands['RE']
        return (RE - R) / (RE + R)


    def _VIopt(self):
        """
          Indices_name:
          Optimal Vegetation Index (VIopt)
          article:
          Reyniers M., Walvoort DJJ, De Baardemaaker J. A linear model to predict with a multi-spectral radiometer the amount of nitrogen in winter wheat. International Journal of Remote Sensing. 2006; 27(19): 4159–4179. DOI: 10.1080/01431160600791650
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return 1.45 * (NIR ** 2 + 1) / (R + 0.45)


    def _WBI(self):
        """
          Indices_name:
          Water Body Index  (WBI)
          article:
          Domenech E, Mallet C. Change detection in high-resolution land use/land cover geodatabases (at object level). EuroSDR Official Publication. 2014; 64.
        """
        R = self.bands['R']
        B = self.bands['B']
        return (B - R) / (B + R)


    def _WDRVI(self):
        """
          Indices_name:
          Wide Dynamic RangeVegetation Index (WDRVI)
          article:
          Gitelson AA. Wide dynamic range vegetation index for remote quantification of biophysical characteristics of vegetation. Journal of Plant Physiology. 2004; 161(2): 165–173. DOI: 10.1078/0176-1617-01176
        """
        R = self.bands['R']
        NIR = self.bands['NIR']
        return (0.15 * NIR - R) / (0.15 * NIR + R)


    def _WI(self):
        """
          Indices_name:
          Woebbecke Index (WI)
          article:
          Ji Y, Liu Z, Liu R, Wang Z, Zong X, Yang T. High-throughput phenotypic traits estimation of faba bean based on machine learning and drone-based multimodal data. Computers and Electronics in Agriculture. 2024; 227(Part 2): 109584. DOI: 10.1016/j.compag.2024.109584
        """
        R = self.bands['R']
        G = self.bands['G']
        B = self.bands['B']
        return (G - B) / (G + R)

# Example: Read band data from an Excel file, calculate vegetation indices, and add them to the table
if __name__ == "__main__":
    # Read the Excel file
    df = pd.read_excel(
        r"C:\Users\expanded_dataset_200.xlsx",
        sheet_name="Sheet1"
    )

    # Check for the presence of specific column names; assign None if missing
    red = df["RED"] if "RED" in df.columns else None
    green = df["GREEN"] if "GREEN" in df.columns else None
    blue = df["BLUE"] if "BLUE" in df.columns else None
    red_edge = df["RedEdge"] if "RedEdge" in df.columns else None
    nir = df["NIR"] if "NIR" in df.columns else None

    # Create the VMID object
    vminstance = VMID(red, green, blue, red_edge, nir, index_list=None)

    # Calculate vegetation indices and concatenate them as new columns
    df1 = vminstance.export_all()
    print(df1)
    df_concat_col = pd.concat([df, df1], axis=1)
    print(df_concat_col)

    # Export the updated table
    df_concat_col.to_excel("indice_data.xlsx", index=False)

