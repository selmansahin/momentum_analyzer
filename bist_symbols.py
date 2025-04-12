def get_bist30_symbols():
    """
    Türkiye borsasındaki (BIST 30) hisse senetlerinin sembollerini döndürür.
    BIST 30, Borsa İstanbul'da işlem gören en yüksek piyasa değerine sahip 30 hissenin yer aldığı endekstir.
    """
    # BIST 30 hisse sembolleri (güncel)
    # BIST hisselerinin Yahoo Finance'deki formatı: SEMBOL.IS
    symbols = [
        'AKBNK.IS', 'ALARK.IS', 'ARCLK.IS', 'ASELS.IS', 
        'BIMAS.IS', 'CCOLA.IS', 'CIMSA.IS', 
         'DOAS.IS', 'DOHOL.IS', 'ECILC.IS', 'EGEEN.IS', 
        'EKGYO.IS', 'ENKAI.IS', 'EREGL.IS', 'FROTO.IS',
        'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 'HEKTS.IS',
         'ISGYO.IS', 'ISCTR.IS', 'KCHOL.IS', 'KONTR.IS', 
         'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 
         'MGROS.IS','ODAS.IS', 'OTKAR.IS',
        'OYAKC.IS', 'PETKM.IS', 'PGSUS.IS','SAHOL.IS', 'SASA.IS',
        'SISE.IS','SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS',
        'TKFEN.IS','TOASO.IS','TSKB.IS', 'TTKOM.IS', 'TTRAK.IS', 'TUPRS.IS', 'ULKER.IS',
        'VAKBN.IS', 'VESBE.IS', 'VESTL.IS', 'YKBNK.IS'        
    ]
    # symbols = [
    #     'AKBNK.IS', 'AKFYE.IS', 'AKGRT.IS', 'AKSA.IS', 'AKSEN.IS', 
    #     'ALARK.IS', 'ALBRK.IS', 'ALFAS.IS', 'ALGYO.IS', 'ALKIM.IS', 
    #     'ARCLK.IS', 'ASELS.IS', 'ASUZU.IS', 'AYGAZ.IS', 'BAGFS.IS', 
    #     'BERA.IS', 'BIENY.IS', 'BIMAS.IS', 'BRISA.IS', 'BRSAN.IS', 
    #     'BRYAT.IS', 'BUCIM.IS', 'CANTE.IS', 'CCOLA.IS', 'CIMSA.IS', 
    #     'DEVA.IS', 'DOAS.IS', 'DOHOL.IS', 'ECILC.IS', 'EGEEN.IS', 
    #     'EKGYO.IS', 'ENKAI.IS', 'EREGL.IS', 'EUPWR.IS', 'FROTO.IS',
    #     'GARAN.IS', 'GENIL.IS', 'GESAN.IS', 'GLYHO.IS', 'GSDHO.IS',
    #     'GUBRF.IS', 'GWIND.IS', 'HALKB.IS', 'HEKTS.IS', 'HLGYO.IS',
    #     'IHLGM.IS', 'ISDMR.IS', 'ISFIN.IS', 'ISGYO.IS', 'ISMEN.IS',
    #     'ISCTR.IS', 'ISMEN.IS', 'KARSN.IS', 'KARTN.IS',
    #     'KCHOL.IS', 'KLMSN.IS', 'KMPUR.IS', 'KONTR.IS', 'KORDS.IS',
    #     'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'LOGO.IS', 'MAVI.IS',
    #     'MGROS.IS', 'NETAS.IS', 'NTHOL.IS', 'ODAS.IS', 'OTKAR.IS',
    #     'OYAKC.IS', 'PENTA.IS', 'PETKM.IS', 'PGSUS.IS', 'QUAGR.IS',
    #     'SAHOL.IS', 'SASA.IS', 'SELEC.IS', 'SISE.IS', 'SKBNK.IS',
    #     'SMART.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS',
    #     'TKFEN.IS', 'TKNSA.IS', 'TLMAN.IS', 'TOASO.IS', 'TRGYO.IS',
    #     'TSKB.IS', 'TTKOM.IS', 'TTRAK.IS', 'TUPRS.IS', 'ULKER.IS',
    #     'VAKBN.IS', 'VERUS.IS', 'VESBE.IS', 'VESTL.IS', 'YKBNK.IS',
    #     'YYLGD.IS', 'ZOREN.IS'
    # ]
    return symbols

def get_bist100_symbols():
    """
    Türkiye borsasındaki (BIST 100) hisse senetlerinin sembollerini döndürür.
    BIST 100, Borsa İstanbul'da işlem gören en yüksek piyasa değerine sahip 100 hissenin yer aldığı endekstir.
    """
    # BIST 100 hisse sembolleri (güncel)
    # BIST hisselerinin Yahoo Finance'deki formatı: SEMBOL.IS
    symbols = [
        'AKBNK.IS', 'AKFYE.IS', 'AKGRT.IS', 'AKSA.IS', 'AKSEN.IS', 
        'ALARK.IS', 'ALBRK.IS', 'ALFAS.IS', 'ALGYO.IS', 'ALKIM.IS', 
        'ARCLK.IS', 'ASELS.IS', 'ASUZU.IS', 'AYGAZ.IS', 'BAGFS.IS', 
        'BERA.IS', 'BIENY.IS', 'BIMAS.IS', 'BRISA.IS', 'BRSAN.IS', 
        'BRYAT.IS', 'BUCIM.IS', 'CANTE.IS', 'CCOLA.IS', 'CIMSA.IS', 
        'DEVA.IS', 'DOAS.IS', 'DOHOL.IS', 'ECILC.IS', 'EGEEN.IS', 
        'EKGYO.IS', 'ENKAI.IS', 'EREGL.IS', 'EUPWR.IS', 'FROTO.IS',
        'GARAN.IS', 'GENIL.IS', 'GESAN.IS', 'GLYHO.IS', 'GSDHO.IS',
        'GUBRF.IS', 'GWIND.IS', 'HALKB.IS', 'HEKTS.IS', 'HLGYO.IS',
        'IHLGM.IS', 'ISDMR.IS', 'ISFIN.IS', 'ISGYO.IS', 'ISMEN.IS',
        'ISCTR.IS', 'ISMEN.IS', 'KARSN.IS', 'KARTN.IS',
        'KCHOL.IS', 'KLMSN.IS', 'KMPUR.IS', 'KONTR.IS', 'KORDS.IS',
        'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'LOGO.IS', 'MAVI.IS',
        'MGROS.IS', 'NETAS.IS', 'NTHOL.IS', 'ODAS.IS', 'OTKAR.IS',
        'OYAKC.IS', 'PENTA.IS', 'PETKM.IS', 'PGSUS.IS', 'QUAGR.IS',
        'SAHOL.IS', 'SASA.IS', 'SELEC.IS', 'SISE.IS', 'SKBNK.IS',
        'SMART.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS',
        'TKFEN.IS', 'TKNSA.IS', 'TLMAN.IS', 'TOASO.IS', 'TRGYO.IS',
        'TSKB.IS', 'TTKOM.IS', 'TTRAK.IS', 'TUPRS.IS', 'ULKER.IS',
        'VAKBN.IS', 'VERUS.IS', 'VESBE.IS', 'VESTL.IS', 'YKBNK.IS',
        'YYLGD.IS', 'ZOREN.IS'
    ]
    return symbols