import sys

class EnterpriseRAGException(Exception):

    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        if exc_tb is not None:

            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.line_number = exc_tb.tb_lineno

        else:
            self.file_name = 'Unknown'
            self.line_number = -1

        super().__init__(self.__str__())

    def __str__(self):

        return(
            f"Error occured in python script"
            f"[{self.file_name}]"
            f"at line [{self.line_number}]"
            f"Error: [{self.error_message}]"
        )