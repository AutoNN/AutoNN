
class TooLowDatasetWarning(Exception):
    '''
    raised when Image dataset is too low with respect
    to the number lowest number of generated parameters
    possible

    '''
    def __init__(self) -> None:
        self.message="""
        Number of datapoints is lower than the
        number of trainable parameters
        """
        super().__init__(self.message) 
    

class IvalidPathError(Exception):
    pass 

class InvalidImageFileError(Exception):
    """Make sure you have only images inside 
        your dataset folder the correct image format"""
    pass 


