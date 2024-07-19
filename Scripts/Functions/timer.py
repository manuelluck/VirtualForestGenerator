from datetime import datetime

class Timer:
    def __init__(self):
        self.time_keeper = {'overall':{'start':datetime.now()}}

    def start_task(self,task):
        self.time_keeper[task] = {'start':datetime.now()}

    def stop_task(self,task):
        if task in self.time_keeper.keys():
            self.time_keeper[task]['stop']      = datetime.now()
            self.time_keeper[task]['runtime']   = self.time_keeper[task]['stop'] - self.time_keeper[task]['start']
        else:
            print('Task not started, available tasks:')
            for key in self.time_keeper.keys():
                print(key)

    def print_task(self,task=None):
        if task in self.time_keeper.keys():
            if 'stop' not in self.time_keeper[task].keys():
                self.time_keeper[task]['stop']  = datetime.now()

            t = self.time_keeper[task]['stop'] - self.time_keeper[task]['start']
            d = t.days
            h, rest = divmod(t.seconds, 3600)
            m, s = divmod(rest, 60)

            self.time_keeper[task]['runtime']       = {'total_seconds':t.total_seconds()}

            if d > 0:
                self.time_keeper[task]['runtime']['days']       = d
                self.time_keeper[task]['runtime']['hours']      = h
                self.time_keeper[task]['runtime']['minutes']    = m
                self.time_keeper[task]['runtime']["seconds"]    = s
            elif h > 0:
                self.time_keeper[task]['runtime']['hours']      = h
                self.time_keeper[task]['runtime']['minutes']    = m
                self.time_keeper[task]['runtime']["seconds"]    = s
            elif m > 0:
                self.time_keeper[task]['runtime']['minutes']    = m
                self.time_keeper[task]['runtime']["seconds"]    = s
            elif s > 0:
                self.time_keeper[task]['runtime']["seconds"]    = s
            else:
                self.time_keeper[task]['runtime']["seconds"]    = 0

            print(f'Task {task} processed in:')

            for key in self.time_keeper[task]['runtime'].keys():
                if key != 'total_seconds':
                    print(f'{key} : {self.time_keeper[task]["runtime"][key]}')
        elif task is None:
            for task in self.time_keeper.keys():
                if 'stop' not in self.time_keeper[task].keys():
                    self.time_keeper[task]['stop']  = datetime.now()

                t = self.time_keeper[task]['stop'] - self.time_keeper[task]['start']
                d = t.days
                h, rest = divmod(t.seconds, 3600)
                m, s = divmod(rest, 60)

                self.time_keeper[task]['runtime']       = {'total_seconds':t.total_seconds()}

                if d > 0:
                    self.time_keeper[task]['runtime']['days'] = d
                    self.time_keeper[task]['runtime']['hours'] = h
                    self.time_keeper[task]['runtime']['minutes'] = m
                    self.time_keeper[task]['runtime']["seconds"] = s
                elif h > 0:
                    self.time_keeper[task]['runtime']['hours'] = h
                    self.time_keeper[task]['runtime']['minutes'] = m
                    self.time_keeper[task]['runtime']["seconds"] = s
                elif m > 0:
                    self.time_keeper[task]['runtime']['minutes'] = m
                    self.time_keeper[task]['runtime']["seconds"] = s
                elif s > 0:
                    self.time_keeper[task]['runtime']["seconds"] = s

                print(f'Task {task} processed in:')
                for key in self.time_keeper[task]['runtime'].keys():
                    if key != 'total_seconds':
                        print(f'{key} : {self.time_keeper[task]["runtime"][key]}')
        else:
            print('Task not started, available tasks:')
            for key in self.time_keeper.keys():
                print(key)
