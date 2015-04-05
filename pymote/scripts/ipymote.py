def start_ipymote():
    import os
    import sys
    if sys.platform.startswith("win32") or sys.platform.startswith("linux2"):
        try:
            os.environ['VIRTUAL_ENV'] = os.environ['PYMOTE_ENV']
            os.environ['IPYTHONDIR'] = os.path.join(os.environ['PYMOTE_ENV'],
                                                    '.ipython')
        except KeyError:
            pass

        from IPython.frontend.terminal.ipapp import TerminalIPythonApp
        app = TerminalIPythonApp.instance()
        app.initialize(argv=['--profile=pymote'])
        app.start()

    #TODO: support other platforms
    #elif sys.platform.startswith("darwin"):
