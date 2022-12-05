
class TooLowDatasetWarning(Exception):
    '''
    raised when Image dataset is too low with respect
    to the number lowest number of generated parameters
    possible

    '''
    def __init__(self) -> None:
        self.message="""
        Number of datapoints is lower than the number of
        trainable parameters. Try augmenting the dataset first.
        """
        super().__init__(self.message) 
    

class InvalidPathError(Exception):
  
    def __init__(self):
        self.message= '''your path is either None or a filename,
         You are requested to provide a valid path to a directory not a FILE'''
        super().__init__(self.message)

class InvalidImageFileError(Exception):
    """Make sure you have only images inside 
        your dataset folder the correct image format"""
    pass 


class InvalidFolderStructureError(Exception):
    '''
    Make sure all the folders are in the specified 
    format accepted by the function
    '''
    pass