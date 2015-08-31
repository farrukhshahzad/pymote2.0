__author__ = 'farrukh.shahzad'

from celery import shared_task
from celery.app.task import Task
from django.db.models import Count
from django.conf import settings

from ps_comms_base.util import create_logger
from ps_comms_base.stats import writer as stats_writer
from ps_error.models import ExceptionHandler

from commservice.web.models import Command

#test_count = 2

@shared_task(ignore_result=True, name="commservice.web.tasks.monitor_collect_stats") # pylint: disable=E1102
@ExceptionHandler
class MonitorCollectStats(Task):

    writer = None
    #stat_counts = None

    def __init__(self):

        Task.name = "commservice.web.tasks.monitor_collect_stats"
        Task.ignore_result = True

        #test_count += 1
        self.logger = create_logger()
        self.logger.info("Initialize %s: %s " % (Task.name, settings.STATS_WRITER))
        #self.logger.info(old_stats=self.stat_counts)

        self.stat_counts = None

        self.run()

    def get_writer(self):
        self.writer = stats_writer.BlockingWriter(settings.STATS_WRITER)

    def run(self):
        """
        A celerybeat task to write statistics related to commands.
        The change in stats/counts are written to CW and statsd since last run
        (default period of 15 min set in settings.py for this task)

        """
        self.logger.info("Running: %s" % Task.name)
        self.get_writer()

        count_array = Command.objects.values('status').annotate(status_count=Count('status'))
        #convert to simple dict
        counts = {}
        change_counts = {}
        try:
            for item in count_array:
                cmd_state = item['status']
                counts[cmd_state] = item['status_count']
                #if self.stat_counts:
                #    change_counts[cmd_state] = counts[cmd_state] - self.stat_counts[cmd_state]
                self.logger.info("Writing stats to %s" % cmd_state)
                if self.writer:
                     self.writer._writer("command_"+cmd_state, counts[cmd_state])
        except Exception as exc:
            self.logger.error(exc)
        self.logger.info(command_stats=counts, change_stats=change_counts)
        #self.stat_counts = counts

        self.logger.info("Done: %s" % Task.name)
        return change_counts
